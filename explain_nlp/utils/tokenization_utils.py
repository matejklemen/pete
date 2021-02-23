from typing import Union, List, Tuple

import torch
from transformers import BertTokenizer


class BertAlignedTokenizationMixin:
    # TODO: assert isinstance(tokenizer, PreTrainedTokenizerFast) and rework!
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
        curr_res = self.tokenizer.encode_plus(text=proxy_ex0, text_pair=proxy_ex1, is_pretokenized=True,
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