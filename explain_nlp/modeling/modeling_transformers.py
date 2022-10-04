from typing import Union, Optional, List, Tuple

import torch
from transformers import BertTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast, \
    XLMRobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, BertForMaskedLM

from explain_nlp.modeling.modeling_base import InterpretableModel
from explain_nlp.utils.tokenization_utils import TransformersAlignedTokenizationMixin


class InterpretableBertBase(InterpretableModel, TransformersAlignedTokenizationMixin):
    tokenizer: BertTokenizerFast

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
                    # Happens if trying to decode special token with skip_special_tokens=True
                    if not str_token:
                        continue

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
                    allow_truncation: Optional[bool] = True):
        truncation_strategy = "longest_first" if allow_truncation else "do_not_truncate"
        return self.encode_aligned(text_data,
                                   is_split_into_units=is_split_into_units,
                                   truncation_strategy=truncation_strategy)

    @torch.no_grad()
    def score(self, input_ids: torch.Tensor, **kwargs):
        raise NotImplementedError


class InterpretableBertForMaskedLM(InterpretableBertBase):
    def __init__(self, tokenizer_name, model_name, batch_size=8, max_seq_len=64, device="cuda"):
        super().__init__(max_seq_len=max_seq_len, batch_size=batch_size, device=device)
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.model = BertForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.aux_data_keys = ["token_type_ids", "attention_mask"]

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
    def __init__(self, tokenizer_name, model_name, batch_size=8, max_seq_len=64, device="cuda"):
        super().__init__(max_seq_len=max_seq_len, batch_size=batch_size, device=device)
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, return_dict=True).to(self.device)
        self.model.eval()

        self.aux_data_keys = ["token_type_ids", "attention_mask"]

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


class InterpretableRobertaBase(InterpretableModel, TransformersAlignedTokenizationMixin):
    tokenizer: Union[RobertaTokenizerFast, XLMRobertaTokenizerFast]

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
                      return_tokens: bool = False, **kwargs):
        num_ex = len(encoded_data)
        attention_fn = None
        if not take_as_single_sequence:
            attention_mask = kwargs["attention_mask"]
            num_aux = attention_mask.shape[0]

            if num_aux == 1:
                # Assume every sequence has the same attention_mask and token_type_ids
                attention_fn = lambda idx_example: attention_mask[0]
            elif num_aux == num_ex:
                attention_fn = lambda idx_example: attention_mask[idx_example]
            else:
                raise ValueError(f"Auxiliary data ({num_aux} ex.) can't be broadcasted to input shape ({num_ex} ex.). "
                                 f"Either provide a single tensor or one tensor per instance")

        if return_tokens:
            special_token_ids_set = self.special_token_ids
            def decoding_fn(input_ids, **decode_kwargs):
                decoded_ids = []
                for curr_id in input_ids:
                    str_token = self.tokenizer.decode(curr_id, **decode_kwargs).strip()
                    # Happens if trying to decode special token with skip_special_tokens=True
                    if not str_token and curr_id in special_token_ids_set:
                        continue

                    decoded_ids.append(str_token)

                return decoded_ids

        else:
            def decoding_fn(input_ids, **decode_kwargs):
                return self.tokenizer.decode(input_ids, **decode_kwargs).strip()

        decoded_data = []
        for idx_example in range(num_ex):
            if take_as_single_sequence:
                decoded_data.append(decoding_fn(encoded_data[idx_example], skip_special_tokens=skip_special_tokens))
            else:
                curr_attendable = attention_fn(idx_example).bool()
                curr_input_ids = encoded_data[idx_example][curr_attendable]
                sep_positions = torch.flatten(torch.nonzero(curr_input_ids == self.tokenizer.sep_token_id,
                                                            as_tuple=False))

                if sep_positions.shape[0] == 1:
                    # <s> <seq> </s>
                    decoded_data.append(decoding_fn(curr_input_ids[1: -1], skip_special_tokens=skip_special_tokens))
                else:
                    # <s> <seq1> </s></s> <seq2> </s>
                    starts = [1] + (sep_positions[1::2] + 1).tolist()
                    ends = sep_positions[::2].tolist() + [-1]

                    multiple_sequences = []
                    for s, e in zip(starts, ends):
                        multiple_sequences.append(decoding_fn(curr_input_ids[s: e],
                                                              skip_special_tokens=skip_special_tokens))
                    decoded_data.append(tuple(multiple_sequences))

        return decoded_data

    def to_internal(self, text_data: Union[List[str], List[Tuple[str, ...]],
                                           List[List[str]], List[Tuple[List[str], ...]]],
                    is_split_into_units: Optional[bool] = False,
                    allow_truncation: Optional[bool] = True):
        truncation_strategy = "longest_first" if allow_truncation else "do_not_truncate"
        return self.encode_aligned(text_data,
                                   is_split_into_units=is_split_into_units,
                                   truncation_strategy=truncation_strategy)

    @torch.no_grad()
    def score(self, input_ids: torch.Tensor, **kwargs):
        raise NotImplementedError


class InterpretableRobertaForSequenceClassification(InterpretableRobertaBase):
    def __init__(self, tokenizer_name, model_name, batch_size=8, max_seq_len=64,
                 device="cuda"):
        super().__init__(max_seq_len=max_seq_len, batch_size=batch_size, device=device)
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True).to(self.device)
        self.model.eval()

        self.aux_data_keys = ["attention_mask"]

    @torch.no_grad()
    def score(self, input_ids: torch.Tensor, **kwargs):
        # If a single example is provided, broadcast it to input size, otherwise assume correct shapes are provided
        if kwargs["attention_mask"].shape[0] != input_ids.shape[0]:
            aux_data = {additional_arg: kwargs[additional_arg].repeat((input_ids.shape[0], 1))
                        for additional_arg in ["attention_mask"]}
        else:
            aux_data = {additional_arg: kwargs[additional_arg]
                        for additional_arg in ["attention_mask"]}

        num_total_batches = (input_ids.shape[0] + self.batch_size - 1) // self.batch_size
        probas = torch.zeros((input_ids.shape[0], self.model.config.num_labels))
        for idx_batch in range(num_total_batches):
            s_b, e_b = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
            curr_input_ids = input_ids[s_b: e_b].to(self.device)
            res = self.model(curr_input_ids, **{k: v[s_b: e_b].to(self.device) for k, v in aux_data.items()})

            probas[s_b: e_b] = torch.softmax(res["logits"], dim=-1)

        return probas


class InterpretableXLMRobertaForSequenceClassification(InterpretableRobertaForSequenceClassification):
    def __init__(self, tokenizer_name, model_name, batch_size=8, max_seq_len=64,
                 device="cuda"):
        InterpretableRobertaBase.__init__(self, max_seq_len=max_seq_len, batch_size=batch_size, device=device)
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name

        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(tokenizer_name, add_prefix_space=True)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(model_name, return_dict=True).to(self.device)
        self.model.eval()

        self.aux_data_keys = ["attention_mask"]
