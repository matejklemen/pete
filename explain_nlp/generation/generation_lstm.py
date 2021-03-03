from typing import Dict, List, Union, Tuple, Optional

import torch

from explain_nlp.custom_modules.contextual_bilstm import ContextualBiLSTMSubwordTokenizer, ContextualBiLSTM
from explain_nlp.generation.generation_base import SampleGenerator


class ContextualBiLSTMLMGenerator(SampleGenerator):
    def __init__(self, tokenizer_name, model_name, max_seq_len, batch_size=8, device="cuda",
                 strategy="top_k", top_p=0.9, top_k=5, threshold=0.1,
                 monte_carlo_dropout: Optional[bool] = False,
                 allowed_values: Optional[List[torch.Tensor]] = None):
        super().__init__(max_seq_len=max_seq_len, batch_size=batch_size, device=device,
                         strategy=strategy, top_p=top_p, top_k=top_k, threshold=threshold)

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

    @property
    def mask_token(self) -> str:
        return self.tokenizer.mask_token

    @property
    def mask_token_id(self) -> int:
        return self.tokenizer.mask_token_id

    def convert_ids_to_tokens(self, ids: torch.Tensor) -> List[List[str]]:
        return [self.tokenizer.convert_ids_to_tokens(curr_ids) for curr_ids in ids.tolist()]

    def from_internal(self, encoded_data, skip_special_tokens: bool = True, take_as_single_sequence: bool = False,
                      **kwargs):
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

        decoded_data = []
        for idx_example in range(num_ex):
            if take_as_single_sequence:
                decoded_data.append(self.tokenizer.decode(encoded_data[idx_example], skip_special_tokens=skip_special_tokens))
            else:
                curr_attendable = attention_fn(idx_example).bool()
                curr_token_types = token_type_fn(idx_example)[curr_attendable]
                curr_input_ids = encoded_data[idx_example][curr_attendable]

                seq_ids, tokens_in_seq = torch.unique(curr_token_types, return_counts=True)
                bins = torch.cumsum(tokens_in_seq, dim=0)
                if seq_ids.shape[0] == 1:
                    decoded_data.append(self.tokenizer.decode(curr_input_ids[1: tokens_in_seq[0]],
                                                              skip_special_tokens=skip_special_tokens))
                else:
                    bins = [1] + bins.tolist()
                    multiple_sequences = []
                    for s, e in zip(bins, bins[1:]):
                        multiple_sequences.append(self.tokenizer.decode(curr_input_ids[s: e - 1],
                                                                        skip_special_tokens=skip_special_tokens))
                    decoded_data.append(tuple(multiple_sequences))

        return decoded_data

    def from_internal_precise(self, encoded_data, skip_special_tokens=True):
        converted = {
            "decoded_data": [],
            "is_continuation": []
        }
        for idx_example in range(encoded_data.shape[0]):
            curr_example = encoded_data[idx_example]
            sep_tokens = torch.nonzero(curr_example == self.tokenizer.sep_token_id, as_tuple=False)
            end = int(sep_tokens[-1])

            processed_example, is_continuation = [], []
            for el in curr_example:
                if skip_special_tokens and el.item() in self.special_tokens_set:
                    processed_example.append("")
                    is_continuation.append(False)
                    continue

                str_tok = self.tokenizer.convert_ids_to_tokens(el.item())
                if str_tok.startswith("##"):
                    processed_example.append(str_tok[2:])
                    is_continuation.append(True)
                else:
                    processed_example.append(str_tok)
                    is_continuation.append(False)

            # Multiple sequences present: [CLS] <seq1> [SEP] <seq2> [SEP] -> (<seq1>, <seq2>)
            if sep_tokens.shape[0] >= 2:
                bnd = int(sep_tokens[0])
                converted["decoded_data"].append((processed_example[1: bnd],
                                                  processed_example[bnd + 1: end]))
                converted["is_continuation"].append((is_continuation[1: bnd],
                                                     is_continuation[bnd + 1: end]))
            else:
                converted["decoded_data"].append(processed_example[1: end])
                converted["is_continuation"].append(is_continuation[1: end])

        return converted

    def to_internal(self, text_data: List[Union[str, Tuple[str, ...]]]) -> Dict:
        res = self.tokenizer.batch_encode_plus(text_data, return_special_tokens_mask=True, return_tensors="pt",
                                               padding="max_length", max_length=self.max_seq_len,
                                               truncation="longest_first")
        formatted_res = {
            "input_ids": res["input_ids"],
            "perturbable_mask": torch.logical_not(res["special_tokens_mask"]),
            "aux_data": {
                "token_type_ids": res["token_type_ids"],
                "attention_mask": res["attention_mask"]
            }
        }

        return formatted_res

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
    NUM_SAMPLES = 10
    sequence = ("A patient is being worked on by doctors and nurses", "A man is sleeping.")
    generator = ContextualBiLSTMLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/contextual_bilstm_lm_snli",
                                            model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/contextual_bilstm_lm_snli",
                                            max_seq_len=41, batch_size=8, device="cpu", strategy="top_p", top_p=0.95,
                                            monte_carlo_dropout=False)

    encoded = generator.to_internal([sequence])
    generated_ids = generator.generate_masked_samples(input_ids=encoded["input_ids"].repeat((NUM_SAMPLES, 1)),
                                                      generation_mask=encoded["perturbable_mask"].repeat((NUM_SAMPLES, 1)))
    generated_text = generator.from_internal(generated_ids, **encoded["aux_data"])
    for ex in generated_text:
        print(ex)

