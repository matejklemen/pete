from typing import List, Set

import torch
from transformers import PreTrainedTokenizerFast

from explain_nlp.utils import EncodingException


class TransformersAlignedTokenizationMixin:
    tokenizer: PreTrainedTokenizerFast
    max_seq_len: int

    new_word_offset: int = 0
    aux_data_keys: List[str]
    # Additional special tokens, that we want to ignore when aligning words
    additional_special_token_ids: Set[int] = set()

    def encode_aligned(self, text_data, is_split_into_units=False, truncation_strategy="do_not_truncate"):
        num_examples = len(text_data)
        res = self.tokenizer.batch_encode_plus(text_data,
                                               is_split_into_words=is_split_into_units,
                                               return_special_tokens_mask=True, return_tensors="pt",
                                               padding="max_length", max_length=self.max_seq_len,
                                               truncation=truncation_strategy)
        if truncation_strategy == "do_not_truncate" and res["input_ids"].shape[1] > self.max_seq_len:
            raise EncodingException(f"Truncation strategy is 'do_not_truncate', but the encoded data is "
                                    f"longer than allowed ({res['input_ids'].shape[1]} > {self.max_seq_len}). "
                                    f"Either set a different strategy or increase max_seq_len.")

        formatted_res = {
            "input_ids": res["input_ids"],
            "perturbable_mask": torch.logical_not(res["special_tokens_mask"]),
            "aux_data": {k: res[k] for k in self.aux_data_keys}
        }

        if is_split_into_units:
            all_word_ids = []
            for idx_example in range(num_examples):
                idx_seq, word_ids = 0, []
                if isinstance(text_data[idx_example], tuple):
                    cumulative_len = [0]
                    for i, curr in enumerate(text_data[idx_example]):
                        cumulative_len.append(len(curr) + cumulative_len[i - 1])
                else:
                    cumulative_len = [0, len(text_data[idx_example])]

                for curr_id, curr_word_id in zip(res["input_ids"][idx_example], res.word_ids(batch_index=idx_example)):
                    # Special tokens that are not present in the input (e.g. CLS, SEP, PAD)
                    if curr_word_id is None:
                        word_ids.append(-1)
                        idx_seq += 1
                    # Special tokens that are present in the input, but are not actual words (e.g. control signal)
                    elif int(curr_id) in self.additional_special_token_ids:
                        word_ids.append(-1)
                    else:
                        word_ids.append(cumulative_len[idx_seq - 1] + curr_word_id)

                all_word_ids.append(word_ids)

            formatted_res["aux_data"]["alignment_ids"] = all_word_ids

        return formatted_res
