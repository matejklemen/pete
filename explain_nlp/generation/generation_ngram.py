import warnings
from typing import Optional, List, Union, Tuple, Dict

import torch

from explain_nlp.custom_modules.ngram import TrigramTokenizer, TrigramMLM, UnigramTokenizer, UnigramModel
from explain_nlp.generation.generation_base import SampleGenerator
from explain_nlp.utils.tokenization_utils import TransformersAlignedTokenizationMixin


class TrigramForMaskedLMGenerator(SampleGenerator, TransformersAlignedTokenizationMixin):
    def __init__(self, tokenizer_name, model_name, max_seq_len, batch_size=8, device="cuda",
                 strategy="top_p", top_p=0.9, top_k=5, threshold=0.1,
                 generate_cover: Optional[bool] = False):
        super().__init__(max_seq_len=max_seq_len, batch_size=batch_size, device=device,
                         strategy=strategy, top_p=top_p, top_k=top_k, threshold=threshold)

        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.generate_cover = generate_cover

        assert self.batch_size > 1 and self.batch_size % 2 == 0
        if device == "cuda":
            warnings.warn("Device 'cuda' is not supported for UnigramLMGenerator, defaulting to cpu")

        self.tokenizer = TrigramTokenizer.from_pretrained(self.tokenizer_name)
        self.generator = TrigramMLM.from_pretrained(self.model_name)

        self.aux_data_keys = ["attention_mask", "token_type_ids"]

    @property
    def mask_token(self) -> str:
        return self.tokenizer.mask_token

    @property
    def mask_token_id(self) -> int:
        return self.tokenizer.mask_token_id

    def from_internal(self, encoded_data, skip_special_tokens: bool = True, take_as_single_sequence: bool = False,
                      return_tokens=False, **kwargs):
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
                    is_split_into_units: Optional[bool] = False) -> Dict:
        return self.encode_aligned(text_data, is_split_into_units=is_split_into_units)


class UnigramLMGenerator(SampleGenerator, TransformersAlignedTokenizationMixin):
    def __init__(self, tokenizer_name, model_name, max_seq_len, batch_size=8, device="cuda",
                 strategy="top_k", top_p=0.9, top_k=5, threshold=0.1):
        super().__init__(max_seq_len=max_seq_len, batch_size=batch_size, device=device,
                         strategy=strategy, top_p=top_p, top_k=top_k, threshold=threshold)

        if device == "cuda":
            warnings.warn("Device 'cuda' is not supported for UnigramLMGenerator, defaulting to cpu")

        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.tokenizer = UnigramTokenizer.from_pretrained(self.tokenizer_name)
        self.generator = UnigramModel.from_pretrained(self.model_name)

        self.aux_data_keys = ["attention_mask", "token_type_ids"]
        self.special_tokens_set = set(self.tokenizer.all_special_ids)

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
                    is_split_into_units: Optional[bool] = False) -> Dict:
        return self.encode_aligned(text_data, is_split_into_units=is_split_into_units)

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

        _generation_mask = generation_mask.bool()
        num_sampled_values = int(torch.sum(_generation_mask))

        probas = torch.exp(self.generator.logprobas)
        sampled_values = torch.multinomial(probas, num_samples=num_sampled_values, replacement=True)
        eff_input_ids[_generation_mask] = sampled_values

        return eff_input_ids


class PositionalUnigramLMGenerator(UnigramLMGenerator):
    def __init__(self, tokenizer_name, model_name, max_seq_len, batch_size=8, device="cuda",
                 strategy="top_k", top_p=0.9, top_k=5, threshold=0.1):
        super().__init__(tokenizer_name=tokenizer_name, model_name=model_name,
                         max_seq_len=max_seq_len, batch_size=batch_size, device=device,
                         strategy=strategy, top_p=top_p, top_k=top_k, threshold=threshold)
        assert self.generator.max_length is not None
        assert max_seq_len <= self.generator.max_length

    def generate_masked_samples(self, input_ids: torch.Tensor,
                                generation_mask: torch.Tensor,
                                **generation_kwargs):
        num_samples = generation_mask.shape[0]
        num_features = input_ids.shape[1]

        assert num_features <= self.generator.max_length
        if input_ids.shape[0] != 1 and input_ids.shape[0] != num_samples:
            raise ValueError(f"input_ids ({input_ids.shape[0]} examples) can't be broadcasted to shape of "
                             f"generation mask ({generation_mask.shape[0]} examples)")

        eff_input_ids = input_ids
        if input_ids.shape[0] == 1:
            eff_input_ids = input_ids.repeat((num_samples, 1))

        _generation_mask = generation_mask.bool()
        _indexer = torch.arange(num_samples)
        probas = torch.exp(self.generator.logprobas)

        for idx_position in range(num_features):
            curr_probas = probas[idx_position]
            num_sampled_values = int(torch.sum(_generation_mask[:, idx_position]))
            if num_sampled_values == 0:
                continue

            sampled_values = torch.multinomial(curr_probas, num_samples=num_sampled_values, replacement=True)
            eff_input_ids[_generation_mask[:, idx_position], idx_position] = sampled_values

        return eff_input_ids


class UnigramControlledLMGenerator(UnigramLMGenerator):
    def __init__(self, tokenizer_name, model_name, control_labels: List[str], max_seq_len,
                 batch_size=8, device="cuda", strategy="top_k", top_p=0.9, top_k=5, threshold=0.1,
                 label_weights: Optional[List] = None):
        super().__init__(tokenizer_name=tokenizer_name, model_name=model_name,
                         max_seq_len=max_seq_len, batch_size=batch_size, device=device,
                         strategy=strategy, top_p=top_p, top_k=top_k, threshold=threshold)

        assert all(curr_control in self.tokenizer.all_special_tokens for curr_control in control_labels)
        self.control_labels = torch.tensor(self.tokenizer.encode(control_labels, add_special_tokens=False))
        self.control_labels_str = control_labels

        self.label_weights = label_weights
        if self.label_weights is None:
            self.label_weights = [1.0] * len(self.control_labels)
        self.label_weights = torch.tensor(self.label_weights)
        self.label_weights /= torch.sum(self.label_weights)

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

        control_labels = generation_kwargs["control_labels"]
        present_labels = set(control_labels)
        idx_label = torch.tensor(list(map(lambda str_class: self.generator.classes[str_class], control_labels)))

        _generation_mask = generation_mask.bool()
        probas = torch.exp(self.generator.logprobas)

        for curr_label in present_labels:
            label_mask = (idx_label == self.generator.classes[curr_label]).unsqueeze(1)
            label_generation_mask = torch.logical_and(_generation_mask, label_mask)
            num_sampled_values = int(torch.sum(label_generation_mask))

            sampled_values = torch.multinomial(probas[self.generator.classes[curr_label]],
                                               num_samples=num_sampled_values, replacement=True)
            eff_input_ids[label_generation_mask] = sampled_values

        return eff_input_ids


if __name__ == "__main__":
    NUM_SAMPLES = 5
    sequence = ("A patient is being worked on by doctors and nurses", "A man is sleeping.")
    generator = PositionalUnigramLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/positional_unigram_lm_snli",
                                             model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/positional_unigram_lm_snli",
                                             max_seq_len=41, batch_size=8, device="cpu", strategy="top_p", top_p=0.95)

    encoded = generator.to_internal([sequence])
    generated_ids = generator.generate_masked_samples(input_ids=encoded["input_ids"].repeat((NUM_SAMPLES, 1)),
                                                      generation_mask=encoded["perturbable_mask"].repeat((NUM_SAMPLES, 1)))
    generated_text = generator.from_internal(generated_ids, **encoded["aux_data"])
    for ex in generated_text:
        print(ex)
