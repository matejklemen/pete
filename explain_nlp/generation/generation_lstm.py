from typing import Dict, List, Union, Tuple, Optional

import torch

from explain_nlp.custom_modules.autoencoder import LSTMAutoencoderSubwordTokenizer, LSTMAutoencoder
from explain_nlp.custom_modules.contextual_bilstm import ContextualBiLSTMSubwordTokenizer, ContextualBiLSTM
from explain_nlp.generation.generation_base import SampleGenerator
from explain_nlp.utils.tokenization_utils import TransformersAlignedTokenizationMixin


class LSTMConditionallyIndependentGenerator(SampleGenerator, TransformersAlignedTokenizationMixin):
    def __init__(self, tokenizer_name, model_name, max_seq_len, batch_size=8, device="cuda",
                 strategy="top_k", top_p=0.9, top_k=5,
                 allowed_values: Optional[List[torch.Tensor]] = None):
        super().__init__(max_seq_len=max_seq_len, batch_size=batch_size, device=device,
                         strategy=strategy, top_p=top_p, top_k=top_k)

        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.tokenizer = LSTMAutoencoderSubwordTokenizer.from_pretrained(self.tokenizer_name)
        self.generator = LSTMAutoencoder.from_pretrained(self.model_name).to(self.device)

        self.aux_data_keys = ["attention_mask", "token_type_ids"]

    @property
    def mask_token(self) -> str:
        return self.tokenizer.mask_token

    @property
    def mask_token_id(self) -> int:
        return self.tokenizer.mask_token_id

    def from_internal(self, encoded_data, skip_special_tokens: bool = True, take_as_single_sequence: bool = False,
                      return_tokens: bool = False, **kwargs):
        num_ex = len(encoded_data)
        token_type_fn, attention_fn = None, None
        if not take_as_single_sequence:
            token_type_ids = kwargs["token_type_ids"]
            attention_mask = kwargs["attention_mask"]
            num_aux = token_type_ids.shape[0]

            if num_aux == 1:
                # Assume every sequence has the same attention_mask and token_type_ids
                token_type_fn = lambda idx_example: token_type_ids[0]
                attention_fn = lambda idx_example: attention_mask[0]
            elif num_aux == num_ex:
                token_type_fn = lambda idx_example: token_type_ids[idx_example]
                attention_fn = lambda idx_example: attention_mask[idx_example]
            else:
                raise ValueError(f"Auxiliary data ({num_aux} ex.) can't be broadcasted to input shape ({num_ex} ex.). "
                                 f"Either provide a single tensor or one tensor per instance")

        if return_tokens:
            def decoding_fn(input_ids, **decode_kwargs):
                decoded_ids = []
                for curr_id in input_ids:
                    str_token = self.tokenizer.decode(curr_id, **decode_kwargs)
                    decoded_ids.append(str_token[2:] if str_token.startswith("##") else str_token)
                return decoded_ids
        else:
            def decoding_fn(input_ids, **decode_kwargs):
                return self.tokenizer.decode(input_ids, **decode_kwargs)

        decoded_data = []
        for idx_example in range(num_ex):
            if take_as_single_sequence:
                decoded_data.append(decoding_fn(encoded_data[idx_example], skip_special_tokens=skip_special_tokens))
            else:
                curr_attendable = attention_fn(idx_example).bool()
                curr_token_types = token_type_fn(idx_example)[curr_attendable]
                curr_input_ids = encoded_data[idx_example][curr_attendable]

                seq_ids, tokens_in_seq = torch.unique(curr_token_types, return_counts=True)
                bins = torch.cumsum(tokens_in_seq, dim=0)
                if seq_ids.shape[0] == 1:
                    decoded_data.append(decoding_fn(curr_input_ids[1: tokens_in_seq[0]],
                                                    skip_special_tokens=skip_special_tokens))
                else:
                    bins = [1] + bins.tolist()
                    multiple_sequences = []
                    for s, e in zip(bins, bins[1:]):
                        multiple_sequences.append(decoding_fn(curr_input_ids[s: e - 1],
                                                              skip_special_tokens=skip_special_tokens))
                    decoded_data.append(tuple(multiple_sequences))

        return decoded_data

    def to_internal(self, text_data: Union[List[str], List[Tuple[str, ...]],
                                           List[List[str]], List[Tuple[List[str], ...]]],
                    is_split_into_units: Optional[bool] = False,
                    allow_truncation: Optional[bool] = True) -> Dict:
        truncation_strategy = "longest_first" if allow_truncation else "do_not_truncate"
        return self.encode_aligned(text_data,
                                   is_split_into_units=is_split_into_units,
                                   truncation_strategy=truncation_strategy)

    @torch.no_grad()
    def generate_masked_samples(self, input_ids: torch.Tensor,
                                generation_mask: torch.Tensor,
                                **generation_kwargs):
        num_samples = generation_mask.shape[0]

        if input_ids.shape[0] != 1 and input_ids.shape[0] != num_samples:
            raise ValueError(f"input_ids ({input_ids.shape[0]} examples) can't be broadcasted to shape of "
                             f"generation mask ({generation_mask.shape[0]} examples)")

        eff_input_ids = input_ids
        if input_ids.shape[0] == 1:
            eff_input_ids = input_ids.repeat((num_samples, 1))

        input_ids_single = input_ids[[0]]
        attention_single = generation_kwargs["attention_mask"][[0]].bool()
        input_len = torch.sum(attention_single, dim=1)
        max_len = input_ids_single.shape[1]

        embedded_input = self.generator.embedder(input_ids_single)  # [1, max_length, embedding_size]
        features, _ = self.generator.lstm(embedded_input)  # [1, max_length, num_directions * hidden_size]
        decomposed_features = features.view(1, max_len, self.generator.num_directions, self.generator.hidden_size)

        latent_repr = decomposed_features[[0], input_len - 1, 0]  # [1, hidden_size]
        if self.generator.num_directions == 2:
            latent_repr = torch.cat((latent_repr, decomposed_features[[0], 0, 1]), dim=1)  # [1, 2 * hidden_size]

        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for idx_batch in range(num_batches):
            s_b, e_b = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
            curr_input_ids = eff_input_ids[s_b: e_b]
            curr_batch_size = curr_input_ids.shape[0]

            curr_latent_repr = latent_repr.repeat((curr_batch_size, 1))
            curr_latent_repr = torch.dropout(curr_latent_repr, p=self.generator.dropout, train=True)

            for idx_position in range(max_len):
                curr_logits = self.generator.decoders[idx_position](curr_latent_repr)
                curr_probas = torch.softmax(curr_logits, dim=-1)
                curr_preds = torch.multinomial(curr_probas, num_samples=1)[:, 0].cpu()

                curr_input_ids[:, idx_position] = curr_preds

        return eff_input_ids


class ContextualBiLSTMLMGenerator(SampleGenerator, TransformersAlignedTokenizationMixin):
    def __init__(self, tokenizer_name, model_name, max_seq_len, batch_size=8, device="cuda",
                 strategy="top_k", top_p=0.9, top_k=5, monte_carlo_dropout: Optional[bool] = False,
                 allowed_values: Optional[List[torch.Tensor]] = None):
        super().__init__(max_seq_len=max_seq_len, batch_size=batch_size, device=device,
                         strategy=strategy, top_p=top_p, top_k=top_k)

        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.tokenizer = ContextualBiLSTMSubwordTokenizer.from_pretrained(self.tokenizer_name)
        self.generator = ContextualBiLSTM.from_pretrained(self.model_name).to(self.device)

        if monte_carlo_dropout:
            self.generator.train()
        else:
            self.generator.eval()

        self.special_tokens_set = set(self.tokenizer.all_special_ids)

        if allowed_values is not None:
            assert len(allowed_values) == self.max_seq_len
            self.impossible_values_mask = torch.ones((self.max_seq_len, len(self.tokenizer)), dtype=torch.bool)
            for idx_feature, curr_possible in enumerate(allowed_values):
                self.impossible_values_mask[idx_feature, curr_possible] = False

            def mask_impossible(curr_logits, position):
                curr_logits[:, self.impossible_values_mask[position, :]] = -float("inf")
                return curr_logits
        else:
            def mask_impossible(curr_logits, position):
                return curr_logits

        self.allowed_values = allowed_values
        self.mask_impossible = mask_impossible

        self.aux_data_keys = ["attention_mask", "token_type_ids"]

    @property
    def mask_token(self) -> str:
        return self.tokenizer.mask_token

    @property
    def mask_token_id(self) -> int:
        return self.tokenizer.mask_token_id

    def from_internal(self, encoded_data, skip_special_tokens: bool = True, take_as_single_sequence: bool = False,
                      return_tokens: bool = False, **kwargs):
        num_ex = len(encoded_data)
        token_type_fn, attention_fn = None, None
        if not take_as_single_sequence:
            token_type_ids = kwargs["token_type_ids"]
            attention_mask = kwargs["attention_mask"]
            num_aux = token_type_ids.shape[0]

            if num_aux == 1:
                # Assume every sequence has the same attention_mask and token_type_ids
                token_type_fn = lambda idx_example: token_type_ids[0]
                attention_fn = lambda idx_example: attention_mask[0]
            elif num_aux == num_ex:
                token_type_fn = lambda idx_example: token_type_ids[idx_example]
                attention_fn = lambda idx_example: attention_mask[idx_example]
            else:
                raise ValueError(f"Auxiliary data ({num_aux} ex.) can't be broadcasted to input shape ({num_ex} ex.). "
                                 f"Either provide a single tensor or one tensor per instance")

        if return_tokens:
            def decoding_fn(input_ids, **decode_kwargs):
                decoded_ids = []
                for curr_id in input_ids:
                    str_token = self.tokenizer.decode(curr_id, **decode_kwargs)
                    decoded_ids.append(str_token[2:] if str_token.startswith("##") else str_token)
                return decoded_ids
        else:
            def decoding_fn(input_ids, **decode_kwargs):
                return self.tokenizer.decode(input_ids, **decode_kwargs)

        decoded_data = []
        for idx_example in range(num_ex):
            if take_as_single_sequence:
                decoded_data.append(decoding_fn(encoded_data[idx_example], skip_special_tokens=skip_special_tokens))
            else:
                curr_attendable = attention_fn(idx_example).bool()
                curr_token_types = token_type_fn(idx_example)[curr_attendable]
                curr_input_ids = encoded_data[idx_example][curr_attendable]

                seq_ids, tokens_in_seq = torch.unique(curr_token_types, return_counts=True)
                bins = torch.cumsum(tokens_in_seq, dim=0)
                if seq_ids.shape[0] == 1:
                    decoded_data.append(decoding_fn(curr_input_ids[1: tokens_in_seq[0]],
                                                    skip_special_tokens=skip_special_tokens))
                else:
                    bins = [1] + bins.tolist()
                    multiple_sequences = []
                    for s, e in zip(bins, bins[1:]):
                        multiple_sequences.append(decoding_fn(curr_input_ids[s: e - 1],
                                                              skip_special_tokens=skip_special_tokens))
                    decoded_data.append(tuple(multiple_sequences))

        return decoded_data

    def to_internal(self, text_data: Union[List[str], List[Tuple[str, ...]],
                                           List[List[str]], List[Tuple[List[str], ...]]],
                    is_split_into_units: Optional[bool] = False,
                    allow_truncation: Optional[bool] = True) -> Dict:
        truncation_strategy = "longest_first" if allow_truncation else "do_not_truncate"
        return self.encode_aligned(text_data,
                                   is_split_into_units=is_split_into_units,
                                   truncation_strategy=truncation_strategy)

    @torch.no_grad()
    def generate_masked_samples(self, input_ids: torch.Tensor,
                                generation_mask: torch.Tensor,
                                **generation_kwargs):
        num_samples = generation_mask.shape[0]
        num_features = input_ids.shape[1]

        if input_ids.shape[0] != 1 and input_ids.shape[0] != num_samples:
            raise ValueError(f"input_ids ({input_ids.shape[0]} examples) can't be broadcasted to shape of "
                             f"generation mask ({generation_mask.shape[0]} examples)")

        eff_input_ids = input_ids
        if input_ids.shape[0] == 1:
            eff_input_ids = input_ids.repeat((num_samples, 1))

        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for idx_batch in range(num_batches):
            s_b, e_b = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size

            curr_inputs = eff_input_ids[s_b: e_b]
            curr_masked = generation_mask[s_b: e_b]

            curr_batch_size = curr_inputs.shape[0]
            _batch_indexer = torch.arange(curr_batch_size)

            for idx_masked_chunk in range(num_features):
                s_c, e_c = idx_masked_chunk, idx_masked_chunk + 1
                is_feature_masked = curr_masked[:, s_c: e_c]

                if not torch.any(is_feature_masked):
                    continue

                logits = self.generator(input_ids=curr_inputs.to(self.device))["logits"]
                curr_logits = logits[:, s_c, :]
                curr_logits[:, self.tokenizer.sep_token_id] = -float("inf")
                curr_logits = self.mask_impossible(curr_logits, position=s_c)
                curr_logits = self.filtering_strategy(curr_logits)

                probas = torch.softmax(curr_logits, dim=-1)
                preds = torch.multinomial(probas, num_samples=1)[:, 0].cpu()

                curr_inputs[is_feature_masked[:, 0], s_c] = preds[is_feature_masked[:, 0]]

        return eff_input_ids


if __name__ == "__main__":
    NUM_SAMPLES = 32
    sequence = ("A shirtless man skateboarding on a ledge.", "A man without a shirt.")
    generator = LSTMConditionallyIndependentGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bilstm_ae_lm_snli",
                                                      model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bilstm_ae_lm_snli",
                                                      max_seq_len=41, batch_size=1, device="cpu", strategy="top_p", top_p=0.9)

    encoded = generator.to_internal([sequence])
    generated_ids = generator.generate_masked_samples(input_ids=encoded["input_ids"].repeat((NUM_SAMPLES, 1)),
                                                      generation_mask=encoded["perturbable_mask"].repeat((NUM_SAMPLES, 1)),
                                                      **encoded["aux_data"])
    generated_text = generator.from_internal(generated_ids, **encoded["aux_data"])
    for ex in generated_text:
        print(ex)

