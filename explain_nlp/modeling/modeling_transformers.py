from typing import Union, Optional, List, Tuple, Dict

import torch
from transformers import BertTokenizer, BertTokenizerFast, BertForMaskedLM, BertForSequenceClassification

from explain_nlp.modeling.modeling_base import InterpretableModel
from explain_nlp.utils.tokenization_utils import BertAlignedTokenizationMixin


class InterpretableBertBase(InterpretableModel, BertAlignedTokenizationMixin):
    tokenizer: Union[BertTokenizer, BertTokenizerFast]

    @property
    def mask_token(self):
        return self.tokenizer.mask_token

    @property
    def mask_token_id(self):
        return self.tokenizer.mask_token_id

    @property
    def pad_token(self) -> str:
        return self.tokenizer.pad_token

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def special_token_ids(self):
        return set(self.tokenizer.all_special_ids)

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
                if skip_special_tokens and el.item() in self.special_token_ids:
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

    def to_internal(self,
                    text_data: Optional[List[Union[str, Tuple[str, ...]]]] = None,
                    pretokenized_text_data: Optional[Union[
                        List[List[str]],
                        List[Tuple[List[str], ...]]
                    ]] = None):
        """ Convert text into model's representation. If `pretokenized_text_data` is given, the word IDs for each
        subword will be given inside ["aux_data"]["alignment_ids"] (id=-1 if it's an unperturbable token)"""
        if pretokenized_text_data is not None:
            res = {
                "input_ids": [], "perturbable_mask": [],
                "aux_data": {"token_type_ids": [], "attention_mask": [], "alignment_ids": []}
            }

            for curr_example_or_pair in pretokenized_text_data:
                curr_res = self.tokenize_aligned(curr_example_or_pair)

                res["input_ids"].append(curr_res["input_ids"])
                res["perturbable_mask"].append(curr_res["perturbable_mask"])
                res["aux_data"]["token_type_ids"].append(curr_res["aux_data"]["token_type_ids"])
                res["aux_data"]["attention_mask"].append(curr_res["aux_data"]["attention_mask"])
                res["aux_data"]["alignment_ids"].append(curr_res["aux_data"]["alignment_ids"])

            res["input_ids"] = torch.cat(res["input_ids"])
            res["perturbable_mask"] = torch.cat(res["perturbable_mask"])
            res["aux_data"]["token_type_ids"] = torch.cat(res["aux_data"]["token_type_ids"])
            res["aux_data"]["attention_mask"] = torch.cat(res["aux_data"]["attention_mask"])
            res["aux_data"]["alignment_ids"] = torch.tensor(res["aux_data"]["alignment_ids"])
        elif text_data is not None:
            res = self.tokenizer.batch_encode_plus(text_data, return_special_tokens_mask=True, return_tensors="pt",
                                                   padding="max_length", max_length=self.max_seq_len,
                                                   truncation="longest_first")
            res = {
                "input_ids": res["input_ids"],
                "perturbable_mask": torch.logical_not(res["special_tokens_mask"]),
                "aux_data": {
                    "token_type_ids": res["token_type_ids"],
                    "attention_mask": res["attention_mask"]
                }
            }
        else:
            raise ValueError("One of 'text_data' or 'pretokenized_text_data' must be given")

        return res

    def words_to_internal(self, pretokenized_text_data: List[Union[List[str], Tuple[List[str], ...]]]) -> Dict:
        """ Convert examples into model's representation, keeping the word boundaries intact.

        Args:
        -----
        text_data:
            Pre-tokenized examples or example pairs
        """
        ret_dict = {
            "words": [],  # text data, augmented with any additional control tokens (used to align importances)
            "input_ids": [],  # list of encoded token subwords for each example/example pair; type: List[List[List[int]]
            # word-level annotations
            "perturbable_mask": [],
            "aux_data": {"token_type_ids": [], "attention_mask": []}
        }

        for curr_example_or_pair in pretokenized_text_data:
            res = self.tokenize_aligned(curr_example_or_pair,
                                        return_raw=True,
                                        group_words=True)

            ret_dict["words"].append(res["words"])
            ret_dict["input_ids"].append(res["input_ids"])
            ret_dict["perturbable_mask"].append(res["perturbable_mask"])
            ret_dict["aux_data"]["token_type_ids"].append(res["aux_data"]["token_type_ids"])
            ret_dict["aux_data"]["attention_mask"].append(res["aux_data"]["attention_mask"])

        ret_dict["perturbable_mask"] = torch.cat(ret_dict["perturbable_mask"])
        ret_dict["aux_data"]["token_type_ids"] = torch.cat(ret_dict["aux_data"]["token_type_ids"])
        ret_dict["aux_data"]["attention_mask"] = torch.cat(ret_dict["aux_data"]["attention_mask"])

        return ret_dict

    def convert_ids_to_tokens(self, ids):
        return [self.tokenizer.convert_ids_to_tokens(curr_ids) for curr_ids in ids.tolist()]

    @torch.no_grad()
    def score(self, input_ids: torch.Tensor, **kwargs):
        raise NotImplementedError


class InterpretableBertForMaskedLM(InterpretableBertBase):
    def __init__(self, tokenizer_name, model_name, batch_size=8, max_seq_len=64, max_words: Optional[int] = 16,
                 device="cuda"):
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_words = max_words

        assert device in ["cpu", "cuda"]
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Device is set to 'cuda', but no CUDA device could be found. If you want to run the model "
                             "on CPU, set device to 'cpu'")
        self.device = torch.device(device)

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model = BertForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, input_ids: torch.Tensor, **kwargs):
        raise NotImplementedError("This function should get overriden by `set_token_scorer`")

    def set_token_scorer(self, idx_token):
        @torch.no_grad()
        def curr_score(input_ids: torch.Tensor, **kwargs):
            # If a single example is provided, broadcast it to input size, otherwise assume correct shapes are provided
            if kwargs["token_type_ids"].shape[0] != input_ids.shape[0]:
                aux_data = {additional_arg: kwargs[additional_arg].repeat((input_ids.shape[0], 1))
                            for additional_arg in ["token_type_ids", "attention_mask"]}
            else:
                aux_data = {additional_arg: kwargs[additional_arg]
                            for additional_arg in ["token_type_ids", "attention_mask"]}

            # Mask the currently predicted token
            # TODO: should attention_mask be 0 for [MASK] token? Currently, the transformers library doesn't do so
            input_ids[:, idx_token] = self.tokenizer.mask_token_id

            num_total_batches = (input_ids.shape[0] + self.batch_size - 1) // self.batch_size
            probas = torch.zeros((input_ids.shape[0], self.tokenizer.vocab_size))
            for idx_batch in range(num_total_batches):
                s_b, e_b = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
                curr_input_ids = input_ids[s_b: e_b].to(self.device)
                res = self.model(curr_input_ids, **{k: v[s_b: e_b].to(self.device) for k, v in aux_data.items()},
                                 return_dict=True)

                tmpo = res["logits"][:, idx_token, :]
                probas[s_b: e_b] = torch.softmax(tmpo, dim=-1)

            return probas

        self.score = curr_score


class InterpretableBertForSequenceClassification(InterpretableBertBase):
    def __init__(self, tokenizer_name, model_name, batch_size=8, max_seq_len=64, max_words: Optional[int] = 16,
                 device="cuda"):
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_words = max_words

        assert device in ["cpu", "cuda"]
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Device is set to 'cuda', but no CUDA device could be found. If you want to run the model "
                             "on CPU, set device to 'cpu'")
        self.device = torch.device(device)

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, return_dict=True).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, input_ids: torch.Tensor, **kwargs):
        # If a single example is provided, broadcast it to input size, otherwise assume correct shapes are provided
        if kwargs["token_type_ids"].shape[0] != input_ids.shape[0]:
            aux_data = {additional_arg: kwargs[additional_arg].repeat((input_ids.shape[0], 1))
                        for additional_arg in ["token_type_ids", "attention_mask"]}
        else:
            aux_data = {additional_arg: kwargs[additional_arg]
                        for additional_arg in ["token_type_ids", "attention_mask"]}

        num_total_batches = (input_ids.shape[0] + self.batch_size - 1) // self.batch_size
        probas = torch.zeros((input_ids.shape[0], self.model.config.num_labels))
        for idx_batch in range(num_total_batches):
            s_b, e_b = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
            curr_input_ids = input_ids[s_b: e_b].to(self.device)
            res = self.model(curr_input_ids, **{k: v[s_b: e_b].to(self.device) for k, v in aux_data.items()})

            probas[s_b: e_b] = torch.softmax(res["logits"], dim=-1)

        return probas
