from typing import Union, List, Tuple, Set

import torch
from transformers import BertTokenizer, PreTrainedTokenizerFast

from explain_nlp.utils import EncodingException


class BertAlignedTokenizationMixin:
    tokenizer: BertTokenizer
    max_seq_len: int
    max_words: int

    def tokenize_aligned(self, curr_example_or_pair: Union[List[str], Tuple[List[str], ...]],
                         return_raw=False, group_words=False):
        """Note: alignment_ids might not be consecutive as some IDs can get truncated! """
        combine_list = list.append if group_words else list.extend  # in-place extension of list with another list
        MAX_LEN = self.max_words if group_words else self.max_seq_len
        is_text_pair = isinstance(curr_example_or_pair, tuple)

        raw_example = []

        ex0, ex1 = curr_example_or_pair if is_text_pair else (curr_example_or_pair, None)
        formatted_ex0, formatted_ex1 = ([], []) if is_text_pair else ([], None)
        word_ids_0, word_ids_1 = ([], []) if is_text_pair else ([], None)
        global_word_id = 0

        for idx_word, curr_word in enumerate(ex0, start=global_word_id):
            curr_subwords = self.tokenizer.encode(curr_word, add_special_tokens=False)
            combine_list(formatted_ex0, curr_subwords)
            word_ids_0.extend([idx_word] * len(curr_subwords))

        global_word_id += len(ex0)

        if is_text_pair:
            for idx_word, curr_word in enumerate(ex1, start=global_word_id):
                curr_subwords = self.tokenizer.encode(curr_word, add_special_tokens=False)
                combine_list(formatted_ex1, curr_subwords)
                word_ids_1.extend([idx_word] * len(curr_subwords))

            global_word_id += len(ex0)

        proxy_ex0 = ["a"] * len(formatted_ex0)
        proxy_ex1 = ["b"] * len(formatted_ex1) if is_text_pair else None

        curr_seq_len = len(formatted_ex0) + (len(formatted_ex1) if is_text_pair else 0)
        num_special_tokens = 3 if is_text_pair else 2
        curr_res = self.tokenizer.encode_plus(text=proxy_ex0, text_pair=proxy_ex1, is_split_into_words=True,
                                              return_special_tokens_mask=True, return_tensors="pt",
                                              padding="max_length", max_length=MAX_LEN,
                                              truncation="longest_first")
        proxy_ex0, proxy_ex1, _ = self.tokenizer.truncate_sequences(ids=proxy_ex0, pair_ids=proxy_ex1,
                                                                    truncation_strategy="longest_first",
                                                                    num_tokens_to_remove=((curr_seq_len + num_special_tokens) - MAX_LEN))

        # [CLS] <seq1> [SEP] [<seq2> [SEP]]
        formatted_example = []
        formatted_example.append([self.tokenizer.cls_token_id] if group_words else self.tokenizer.cls_token_id)
        formatted_example.extend(formatted_ex0[:len(proxy_ex0)])
        formatted_example.append([self.tokenizer.sep_token_id] if group_words else self.tokenizer.sep_token_id)
        word_ids = [-1] + word_ids_0[:len(proxy_ex0)] + [-1]

        if return_raw:
            raw_example.append(self.tokenizer.cls_token)
            raw_example.extend(ex0[:len(proxy_ex0)])
            raw_example.append(self.tokenizer.sep_token)

        if is_text_pair:
            formatted_example.extend(formatted_ex1[:len(proxy_ex1)])
            formatted_example.append([self.tokenizer.sep_token_id] if group_words else self.tokenizer.sep_token_id)

            word_ids.extend(word_ids_1[:len(proxy_ex1)])
            word_ids.append(-1)

            if return_raw:
                raw_example.extend(ex1[:len(proxy_ex1)])
                raw_example.append(self.tokenizer.sep_token)

        word_ids += [-1] * (self.max_seq_len - len(word_ids))
        PAD_TOKEN = [self.tokenizer.pad_token_id] if group_words else self.tokenizer.pad_token_id
        formatted_example.extend([PAD_TOKEN for _ in range(MAX_LEN - len(formatted_example))])

        ret_dict = {
            "input_ids": formatted_example if group_words else torch.tensor([formatted_example]),
            "perturbable_mask": torch.logical_not(curr_res["special_tokens_mask"]),
            "aux_data": {
                "token_type_ids": curr_res["token_type_ids"],
                "attention_mask": curr_res["attention_mask"]
            }
        }

        if return_raw:
            raw_example.extend([self.tokenizer.pad_token for _ in range(MAX_LEN - len(raw_example))])
            ret_dict["words"] = raw_example

        if not group_words:
            ret_dict["aux_data"]["alignment_ids"] = word_ids

        return ret_dict


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
                                               return_offsets_mapping=is_split_into_units,
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
                idx_word, word_ids = -1, []
                for curr_id, curr_offset in zip(res["input_ids"][idx_example], res["offset_mapping"][idx_example]):
                    s, e = curr_offset
                    # Special token
                    if s == e == 0 or int(curr_id) in self.additional_special_token_ids:
                        word_ids.append(-1)
                    else:
                        if s == self.new_word_offset:
                            idx_word += 1
                        word_ids.append(idx_word)

                all_word_ids.append(word_ids)

            formatted_res["aux_data"]["alignment_ids"] = all_word_ids

        return formatted_res
