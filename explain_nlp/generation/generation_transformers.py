import warnings
from typing import Optional, List, Union, Tuple, Dict

import torch
from transformers import OpenAIGPTLMHeadModel, BertForMaskedLM, RobertaForMaskedLM, XLMRobertaTokenizerFast, \
    XLMRobertaForMaskedLM
from transformers import OpenAIGPTTokenizerFast, BertTokenizerFast, RobertaTokenizerFast

from explain_nlp.generation.generation_base import SampleGenerator
from explain_nlp.generation.decoding import greedy_decoding, top_p_decoding
from explain_nlp.methods.utils import extend_tensor
from explain_nlp.utils.tokenization_utils import TransformersAlignedTokenizationMixin


class TransformersMLMGenerationMixin:
    generator: Union[BertForMaskedLM, RobertaForMaskedLM, XLMRobertaForMaskedLM]
    tokenizer: Union[BertTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast]
    device: torch.device
    filters: List
    batch_size: int

    # keys that are used in model beside input_ids, e.g. "attention_mask"
    aux_data_keys: List[str]

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor,
                 num_samples: Optional[int], **generation_kwargs):
        # Note: currently assuming generation additional data is same for all samples
        eff_input_ids = input_ids
        if eff_input_ids.shape[0] == 1:
            eff_input_ids = eff_input_ids.repeat((num_samples, 1))

        eff_aux_data = {}
        for attr_name in self.aux_data_keys:
            attr_value = generation_kwargs[attr_name]
            if attr_value.shape[0] == 1:
                attr_value = attr_value.repeat((self.batch_size, 1))

            attr_value = attr_value.to(self.device)
            eff_aux_data[attr_name] = attr_value

        num_attrs = eff_input_ids.shape[1]
        perturbable_inds = torch.arange(num_attrs)[perturbable_mask[0]]

        # Shuffle generation order to introduce some variance (needed especially if using greedy decoding)
        weights = torch.zeros_like(eff_input_ids, dtype=torch.float32)
        weights[:, perturbable_inds] = 1
        generation_order = torch.multinomial(weights, num_samples=perturbable_inds.shape[0], replacement=False)

        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for idx_batch in range(num_batches):
            s_b, e_b = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
            curr_input_ids = eff_input_ids[s_b: e_b]  # view
            curr_gen_order = generation_order[s_b: e_b]
            orig_tokens = curr_input_ids.clone()

            curr_batch_size = curr_input_ids.shape[0]
            batch_indexer = torch.arange(curr_batch_size)

            for i in range(curr_gen_order.shape[1]):
                curr_indices = curr_gen_order[:, i]
                curr_input_ids[batch_indexer, curr_indices] = self.tokenizer.mask_token_id

                curr_aux_data = {k: v[: curr_batch_size] for k, v in eff_aux_data.items()}

                logits = self.generator(input_ids=curr_input_ids.to(self.device),
                                        **curr_aux_data)["logits"]

                curr_logits = logits[batch_indexer, curr_indices, :]
                for curr_filter in self.filters:
                    curr_logits = curr_filter(curr_logits, orig_values=orig_tokens[batch_indexer, curr_indices])

                probas = torch.softmax(curr_logits, dim=-1)
                preds = torch.multinomial(probas, num_samples=1)[:, 0].cpu()

                curr_input_ids[batch_indexer, curr_indices] = preds

        return {
            "input_ids": eff_input_ids
        }

    @torch.no_grad()
    def generate_masked_samples(self, input_ids: torch.Tensor, generation_mask: torch.Tensor, **generation_kwargs):
        """ This function can either be used to generate multiple perturbations for single example or single
        perturbation per example"""
        num_samples = generation_mask.shape[0]
        num_features = input_ids.shape[1]

        if input_ids.shape[0] != 1 and input_ids.shape[0] != num_samples:
            raise ValueError(f"input_ids ({input_ids.shape[0]} examples) can't be broadcasted to shape of "
                             f"generation mask ({generation_mask.shape[0]} examples)")

        eff_input_ids = input_ids
        if input_ids.shape[0] == 1:
            eff_input_ids = input_ids.repeat((num_samples, 1))

        # Note: currently assuming generation additional data is same for all samples
        # Either specific additional data is provided or additional data is same for all samples
        eff_aux_data = {}
        for k in self.aux_data_keys:
            eff_aux_data[k] = generation_kwargs[k]
            if generation_kwargs[k].shape[0] == 1:
                eff_aux_data[k] = generation_kwargs[k].repeat((num_samples, 1))

        mask_size = 1
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        # A chunk = a part that is masked all at once - currently, this is fixed to a single unit
        num_total_chunks = (num_features + mask_size - 1) // mask_size

        for idx_batch in range(num_batches):
            s_b, e_b = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size

            curr_inputs = eff_input_ids[s_b: e_b]
            orig_tokens = curr_inputs.clone()
            curr_masked = generation_mask[s_b: e_b]

            curr_batch_size = curr_inputs.shape[0]
            _batch_indexer = torch.arange(curr_batch_size)

            # Move left to right by a sliding window of width `mask_size`
            for idx_masked_chunk in range(num_total_chunks):
                s_c, e_c = idx_masked_chunk * mask_size, (idx_masked_chunk + 1) * mask_size
                is_feature_masked = curr_masked[:, s_c: e_c]
                curr_mask_size = is_feature_masked.shape[1]

                if not torch.any(is_feature_masked):
                    continue

                curr_inputs[:, s_c: e_c][is_feature_masked] = self.tokenizer.mask_token_id
                curr_aux_data = {k: v[s_b: e_b].to(self.device) for k, v in eff_aux_data.items()}

                logits = self.generator(input_ids=curr_inputs.to(self.device), **curr_aux_data)["logits"]
                for pos in range(curr_mask_size):
                    curr_logits = logits[:, s_c + pos, :]
                    for curr_filter in self.filters:
                        curr_logits = curr_filter(curr_logits, orig_values=orig_tokens[:, s_c + pos])

                    probas = torch.softmax(curr_logits, dim=-1)
                    preds = torch.multinomial(probas, num_samples=1)[:, 0].cpu()

                    curr_inputs[is_feature_masked[:, pos], s_c + pos] = preds[is_feature_masked[:, pos]]

        return eff_input_ids


class TransformersCMLMGenerationMixin(TransformersMLMGenerationMixin):
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor,
                 num_samples: Optional[int], **generation_kwargs):
        # Note: currently assuming generation additional data is same for all samples
        # Make room for control label at start of sequence (at pos. 1)
        extended_input_ids = extend_tensor(input_ids)
        if extended_input_ids.shape[0] == 1:
            extended_input_ids = extended_input_ids.repeat((num_samples, 1))

        extended_pert_mask = extend_tensor(perturbable_mask)
        eff_aux_data = {k: extend_tensor(generation_kwargs[k]) for k in self.aux_data_keys}

        # Control labels are attendable
        eff_aux_data["attention_mask"][:, 1] = 1

        selected_control_labels = self.tokenizer.convert_tokens_to_ids(generation_kwargs["control_labels"])
        selected_control_labels = torch.tensor(selected_control_labels)
        extended_input_ids[:, 1] = selected_control_labels

        generated_res = super().generate(input_ids=extended_input_ids,
                                         perturbable_mask=extended_pert_mask,
                                         num_samples=num_samples, **eff_aux_data)

        # Remove the control label
        valid_tokens = torch.ones_like(extended_pert_mask)
        valid_tokens[0, 1] = False

        return {
            "input_ids": generated_res["input_ids"][:, valid_tokens[0]]
        }

    @torch.no_grad()
    def generate_masked_samples(self, input_ids: torch.Tensor,
                                generation_mask: torch.Tensor,
                                **generation_kwargs):
        num_samples = generation_mask.shape[0]

        eff_input_ids = extend_tensor(input_ids)
        if eff_input_ids.shape[0] != num_samples:
            eff_input_ids = eff_input_ids.repeat((num_samples, 1))
        eff_generation_mask = extend_tensor(generation_mask)

        # Either specific additional data is provided or additional data is same for all samples
        eff_aux_data = {}
        for k in self.aux_data_keys:
            eff_aux_data[k] = generation_kwargs[k]
            if generation_kwargs[k].shape[0] == 1:
                eff_aux_data[k] = generation_kwargs[k].repeat((num_samples, 1))

        eff_aux_data = {k: extend_tensor(v) for k, v in eff_aux_data.items()}
        # Control labels are attendable
        eff_aux_data["attention_mask"][:, 1] = 1
        eff_generation_mask[:, 1] = False

        control_labels = generation_kwargs["control_labels"]
        encoded_control_labels = torch.tensor(self.tokenizer.convert_tokens_to_ids(control_labels))
        eff_input_ids[:, 1] = encoded_control_labels

        all_examples = super().generate_masked_samples(input_ids=eff_input_ids,
                                                       generation_mask=eff_generation_mask,
                                                       **eff_aux_data)

        valid_tokens = torch.ones(eff_input_ids.shape[1], dtype=torch.bool)
        valid_tokens[1] = False

        return all_examples[:, valid_tokens]


# TODO: rework generate, but first think about whether it will follow MLM-style generation or something else
class GPTLMGenerator(SampleGenerator, TransformersAlignedTokenizationMixin):
    def __init__(self, tokenizer_name, model_name, max_seq_len, batch_size=2, device="cuda",
                 strategy="top_p", top_p=0.9, top_k=5):
        super().__init__(max_seq_len=max_seq_len, batch_size=batch_size, device=device,
                         strategy=strategy, top_p=top_p, top_k=top_k)

        self.tokenizer_name = tokenizer_name
        self.model_name = model_name

        self.tokenizer = OpenAIGPTTokenizerFast.from_pretrained(self.tokenizer_name)
        self.generator = OpenAIGPTLMHeadModel.from_pretrained(self.model_name, return_dict=True)
        if self.tokenizer.mask_token_id is None:
            warnings.warn(f"'mask_token' is not set in GPTLMGenerator's tokenizer, temporarily setting it to '<MASK>'")
            self.tokenizer.add_special_tokens({"mask_token": "<MASK>"})
            self.generator.resize_token_embeddings(len(self.tokenizer))

        self.generator = self.generator.to(self.device)
        self.generator.eval()

        # Required to build sequence: <BOS> <seq1> [<SEP> <seq2>] <EOS>
        assert self.tokenizer.bos_token_id is not None
        assert self.tokenizer.sep_token_id is not None
        assert self.tokenizer.eos_token_id is not None

        self.additional_special_token_ids = {self.tokenizer.bos_token_id, self.tokenizer.sep_token_id,
                                             self.tokenizer.eos_token_id}
        self.aux_data_keys = ["attention_mask"]

    @property
    def mask_token(self):
        return self.tokenizer.mask_token

    @property
    def mask_token_id(self):
        return self.tokenizer.mask_token_id

    def from_internal(self, encoded_data, skip_special_tokens: bool = True, take_as_single_sequence: bool = False,
                      return_tokens=False, **kwargs):
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
        for idx_example in range(len(encoded_data)):
            curr_example = encoded_data[idx_example]
            if take_as_single_sequence:
                decoded_data.append(decoding_fn(encoded_data[idx_example], skip_special_tokens=skip_special_tokens))
            else:
                sep_tokens = torch.nonzero(curr_example == self.tokenizer.sep_token_id, as_tuple=False)
                eos_tokens = torch.nonzero(curr_example == self.tokenizer.eos_token_id, as_tuple=False)
                end = int(eos_tokens[-1])

                # Multiple sequences present: <BOS> <seq1> <SEP> <seq2> <EOS> -> (<seq1>, <seq2>)
                if sep_tokens.shape[0] == 1:
                    bnd = int(sep_tokens[0])
                    seq1 = decoding_fn(curr_example[1: bnd], skip_special_tokens=skip_special_tokens)
                    seq2 = decoding_fn(curr_example[bnd + 1: end], skip_special_tokens=skip_special_tokens)
                    decoded_data.append((seq1, seq2))
                else:
                    decoded_data.append(decoding_fn(curr_example[: end], skip_special_tokens=skip_special_tokens))

        return decoded_data

    def to_internal(self, text_data: Union[List[str], List[Tuple[str, ...]],
                                           List[List[str]], List[Tuple[List[str], ...]]],
                    is_split_into_units: Optional[bool] = False,
                    allow_truncation: Optional[bool] = True) -> Dict:
        truncation_strategy = "longest_first" if allow_truncation else "do_not_truncate"
        if is_split_into_units:
            def format_example(curr_text):
                if isinstance(curr_text, list):
                    return [self.tokenizer.bos_token] + curr_text + [self.tokenizer.sep_token]
                else:  # tuple
                    return [self.tokenizer.bos_token] + curr_text[0] + [self.tokenizer.sep_token] + curr_text[1] + [self.tokenizer.eos_token]
        else:
            def format_example(curr_text):
                if isinstance(curr_text, str):
                    return f"{self.tokenizer.bos_token} {curr_text} {self.tokenizer.sep_token}"
                else:  # tuple
                    return f"{self.tokenizer.bos_token} {curr_text[0]} {self.tokenizer.sep_token} {curr_text[1]} {self.tokenizer.eos_token}"

        _text_data = []
        for curr_text in text_data:
            _text_data.append(format_example(curr_text))

        res = self.encode_aligned(_text_data,
                                  is_split_into_units=is_split_into_units,
                                  truncation_strategy=truncation_strategy)
        for idx_example in range(res["input_ids"].shape[0]):
            for idx_feature in range(res["input_ids"].shape[1]):
                curr_input_id = int(res["input_ids"][idx_example, idx_feature])
                res["perturbable_mask"][idx_example, idx_feature] &= \
                    curr_input_id not in self.additional_special_token_ids

        return res

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor, num_samples: int, label=None,
                 **aux_data) -> Dict:
        num_features = int(input_ids.shape[1])

        perturbable_inds = torch.arange(num_features)[perturbable_mask[0]]
        num_perturbable = perturbable_inds.shape[0]
        masked_samples = input_ids.repeat((num_samples, 1))

        if self.top_p is None:
            def decoding_strategy(logits, ensure_diff_from):
                return greedy_decoding(logits, ensure_diff_from)

            # Randomly decide which predictions to take at each step in order to ensure some diversity
            # (otherwise we would get `num_samples` completely identical samples with greedy decoding)
            probas = torch.zeros((num_samples, num_features))
            probas[:, perturbable_inds] = 1 / num_perturbable
            permuted_indices = torch.multinomial(probas, num_samples=num_perturbable)
        else:
            def decoding_strategy(logits, ensure_diff_from):
                return top_p_decoding(logits, self.top_p, ensure_diff_from)

            # Predict last tokens first and first tokens last to make use of as much context as possible
            permuted_indices = perturbable_inds.repeat((num_samples, 1))

        attention_mask = aux_data["attention_mask"]
        generation_data = {
            "attention_mask": attention_mask.repeat((self.batch_size, 1)).to(self.device)
        }

        for idx_chunk in range(num_perturbable):
            curr_masked = permuted_indices[:, idx_chunk]
            num_batches = (num_samples + self.batch_size - 1) // self.batch_size
            for idx_batch in range(num_batches):
                s_batch, e_batch = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
                curr_input_ids = masked_samples[s_batch: e_batch]
                curr_batch_size = curr_input_ids.shape[0]

                res = self.generator(curr_input_ids.to(self.device),
                                     attention_mask=generation_data["attention_mask"][: curr_batch_size],
                                     return_dict=True)

                curr_token_logits = res["logits"][torch.arange(curr_batch_size), curr_masked[s_batch: e_batch] - 1]

                curr_preds = decoding_strategy(curr_token_logits,
                                               ensure_diff_from=None)

                masked_samples[torch.arange(s_batch, s_batch + curr_batch_size),
                               curr_masked[s_batch: e_batch]] = curr_preds[:, 0].cpu()

        # TODO: once reworked, return weights corresponding to probabilities here
        sample_token_weights = torch.ones_like(masked_samples, dtype=torch.float32)

        return {
            "input_ids": masked_samples,
            "weights": sample_token_weights
        }

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

        # Note: currently assuming generation additional data is same for all samples
        eff_aux_data = {k: generation_kwargs[k].repeat((self.batch_size, 1)).to(self.device)
                        for k in ["attention_mask"]}

        mask_size = 1
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        num_total_chunks = (num_features + mask_size - 1) // mask_size

        for idx_batch in range(num_batches):
            s_b, e_b = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size

            curr_inputs = eff_input_ids[s_b: e_b]
            orig_tokens = curr_inputs.clone()
            curr_masked = generation_mask[s_b: e_b]

            curr_batch_size = curr_inputs.shape[0]
            _batch_indexer = torch.arange(curr_batch_size)

            # Move left to right by a sliding window of width `mask_size`; [0] can't be predicted using LM
            for idx_masked_chunk in range(1, num_total_chunks):
                s_c, e_c = idx_masked_chunk * mask_size, (idx_masked_chunk + 1) * mask_size
                is_feature_masked = curr_masked[:, s_c: e_c]
                curr_mask_size = is_feature_masked.shape[1]

                if not torch.any(is_feature_masked):
                    continue

                curr_inputs[:, s_c: e_c][is_feature_masked] = self.tokenizer.mask_token_id
                curr_aux_data = {k: v[:curr_batch_size, :s_c] for k, v in eff_aux_data.items()}

                logits = self.generator(input_ids=curr_inputs[:, :s_c].to(self.device), **curr_aux_data)["logits"]
                for pos in range(curr_mask_size):
                    curr_logits = logits[:, s_c + pos - 1, :]
                    # Special tokens are set in place, no new ones should be predicted
                    curr_logits[:, self.tokenizer.bos_token_id] = -float("inf")
                    curr_logits[:, self.tokenizer.sep_token_id] = -float("inf")
                    curr_logits[:, self.tokenizer.eos_token_id] = -float("inf")

                    for curr_filter in self.filters:
                        curr_logits = curr_filter(curr_logits, orig_values=orig_tokens[:, s_c + pos])

                    probas = torch.softmax(curr_logits, dim=-1)
                    preds = torch.multinomial(probas, num_samples=1)[:, 0].cpu()

                    curr_inputs[is_feature_masked[:, pos], s_c + pos] = preds[is_feature_masked[:, pos]]

        return eff_input_ids


class GPTControlledLMGenerator(GPTLMGenerator):
    def __init__(self, tokenizer_name, model_name, control_labels: List[str], max_seq_len,
                 batch_size=2, device="cuda", strategy="greedy", top_p=0.9, top_k=5,
                 label_weights: Optional[List] = None):
        super().__init__(tokenizer_name=tokenizer_name, model_name=model_name, batch_size=batch_size,
                         max_seq_len=max_seq_len, device=device, top_p=top_p, top_k=top_k, strategy=strategy)

        assert all(curr_control in self.tokenizer.all_special_tokens for curr_control in control_labels)
        self.control_labels = torch.tensor(self.tokenizer.encode(control_labels, add_special_tokens=False))
        self.control_labels_str = control_labels

        self.label_weights = label_weights
        if self.label_weights is None:
            self.label_weights = [1.0] * len(self.control_labels)
        self.label_weights = torch.tensor(self.label_weights)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor, num_samples: int, label: int,
                 **aux_data) -> Dict:
        # Placeholder data which will be overwritten by generated examples
        generated_input_ids = input_ids.repeat((num_samples, 1))

        # Randomly select control labels for examples to be generated and set as attendable
        idx_selected_control_labels = torch.multinomial(self.label_weights, num_samples=num_samples, replacement=True)
        selected_control_labels = self.control_labels[idx_selected_control_labels]
        # Replace BOS token with control label
        generated_input_ids[:, 0] = selected_control_labels

        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for idx_batch in range(num_batches):
            s_b, e_b = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
            curr_input_ids = generated_input_ids[s_b: e_b]  # view
            curr_batch_size = curr_input_ids.shape[0]

            curr_generated_ids = torch.zeros((curr_batch_size, self.max_seq_len), dtype=generated_input_ids.dtype)
            curr_generated_ids[:, :1] = curr_input_ids[:, :1]
            dummy_attention_masks = torch.ones_like(curr_generated_ids)

            is_eos = torch.zeros(curr_batch_size, dtype=torch.bool)
            for idx_step in range(1, self.max_seq_len):
                if torch.all(is_eos):
                    break

                generation_mask = torch.logical_not(is_eos)
                if torch.any(generation_mask):
                    res = self.generator(input_ids=curr_generated_ids[generation_mask, :idx_step].to(self.device),
                                         attention_mask=dummy_attention_masks[generation_mask, :idx_step].to(self.device))
                    logits = res["logits"][:, -1, :]
                    logits = self.filtering_strategy(logits)
                    probas = torch.softmax(logits, dim=-1)

                    preds = torch.multinomial(probas, num_samples=1)
                    curr_generated_ids[generation_mask, idx_step] = preds[:, 0].cpu()

                is_eos = torch.logical_or(is_eos, curr_generated_ids[:, idx_step] == self.tokenizer.eos_token_id)

            generated_input_ids[s_b: e_b] = curr_generated_ids

        generated_input_ids[:, 0] = self.tokenizer.bos_token_id
        generated_input_ids[:, -1] = self.tokenizer.eos_token_id

        return {
            "input_ids": generated_input_ids
        }

    @torch.no_grad()
    def generate_masked_samples(self, input_ids: torch.Tensor,
                                generation_mask: torch.Tensor,
                                **generation_kwargs):
        eff_input_ids = extend_tensor(input_ids)
        eff_generation_mask = extend_tensor(generation_mask)

        # Note: currently assuming generation additional data is same for all samples
        eff_aux_data = {k: generation_kwargs[k].repeat((self.batch_size, 1)).to(self.device)
                        for k in ["attention_mask"]}
        eff_aux_data = {k: extend_tensor(v) for k, v in eff_aux_data.items()}
        # Control labels are attendable
        eff_aux_data["attention_mask"][:, 1] = 1
        eff_generation_mask[:, 1] = False

        control_labels = generation_kwargs["control_labels"]
        encoded_control_labels = self.tokenizer.encode(control_labels, add_special_tokens=False)
        eff_input_ids[:, 1] = torch.tensor(encoded_control_labels)

        all_examples = super().generate_masked_samples(input_ids=eff_input_ids,
                                                       generation_mask=eff_generation_mask,
                                                       **eff_aux_data)
        valid_tokens = torch.ones(eff_input_ids.shape[1], dtype=torch.bool)
        valid_tokens[1] = False

        return all_examples[:, valid_tokens]


class BertForMaskedLMGenerator(SampleGenerator, TransformersMLMGenerationMixin, TransformersAlignedTokenizationMixin):
    def __init__(self, tokenizer_name, model_name, max_seq_len, batch_size=8, device="cuda",
                 strategy="top_k", top_p=0.9, top_k=5, monte_carlo_dropout: Optional[bool] = False):
        super().__init__(max_seq_len=max_seq_len, batch_size=batch_size, device=device,
                         strategy=strategy, top_p=top_p, top_k=top_k)
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name

        assert self.batch_size > 1 and self.batch_size % 2 == 0

        self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer_name)
        self.generator = BertForMaskedLM.from_pretrained(self.model_name, return_dict=True).to(self.device)
        if monte_carlo_dropout:
            self.generator.train()
        else:
            self.generator.eval()

        self.special_tokens_set = set(self.tokenizer.all_special_ids)
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

    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor,
                 num_samples: int, **aux_data) -> Dict:
        return TransformersMLMGenerationMixin.generate(self, input_ids=input_ids, perturbable_mask=perturbable_mask,
                                                       num_samples=num_samples, **aux_data)

    def generate_masked_samples(self, input_ids: torch.Tensor, generation_mask: torch.Tensor, **generation_kwargs):
        return TransformersMLMGenerationMixin.generate_masked_samples(self, input_ids=input_ids,
                                                                      generation_mask=generation_mask,
                                                                      **generation_kwargs)


class BertForControlledMaskedLMGenerator(BertForMaskedLMGenerator, TransformersCMLMGenerationMixin):
    def __init__(self, tokenizer_name, model_name, control_labels: List[str], max_seq_len,
                 batch_size=8, device="cuda", strategy="top_p", top_p=0.9, top_k=5,
                 label_weights: Optional[List] = None, monte_carlo_dropout: Optional[bool] = False):
        super().__init__(tokenizer_name=tokenizer_name, model_name=model_name,
                         batch_size=batch_size, max_seq_len=max_seq_len, device=device,
                         strategy=strategy, top_p=top_p, top_k=top_k, monte_carlo_dropout=monte_carlo_dropout)

        assert all(curr_control in self.tokenizer.all_special_tokens for curr_control in control_labels)
        self.control_labels = torch.tensor(self.tokenizer.encode(control_labels, add_special_tokens=False,
                                                                 is_split_into_words=True))
        self.control_labels_str = control_labels

        # make it impossible to sample control labels (those are set in place)
        def mask_control(logits, **kwargs):
            logits[:, self.control_labels] = -float("inf")
            return logits
        self.filters = [mask_control] + self.filters

        self.label_weights = label_weights
        if self.label_weights is None:
            self.label_weights = [1.0] * len(self.control_labels)
        self.label_weights = torch.tensor(self.label_weights)
        self.label_weights /= torch.sum(self.label_weights)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor,
                 num_samples: Optional[int], **generation_kwargs):
       return TransformersCMLMGenerationMixin.generate(self, input_ids=input_ids, perturbable_mask=perturbable_mask,
                                                       num_samples=num_samples,
                                                       **generation_kwargs)

    @torch.no_grad()
    def generate_masked_samples(self, input_ids: torch.Tensor,
                                generation_mask: torch.Tensor,
                                **generation_kwargs):
        return TransformersCMLMGenerationMixin.generate_masked_samples(self, input_ids=input_ids,
                                                                       generation_mask=generation_mask,
                                                                       **generation_kwargs)


class RobertaForMaskedLMGenerator(SampleGenerator, TransformersMLMGenerationMixin,
                                  TransformersAlignedTokenizationMixin):
    def __init__(self, tokenizer_name, model_name, max_seq_len, batch_size=8, device="cuda",
                 strategy="top_k", top_p=0.9, top_k=5, monte_carlo_dropout: Optional[bool] = False):
        super().__init__(max_seq_len=max_seq_len, batch_size=batch_size, device=device,
                         strategy=strategy, top_p=top_p, top_k=top_k)
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name

        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.tokenizer_name, add_prefix_space=True)
        self.generator = RobertaForMaskedLM.from_pretrained(self.model_name, return_dict=True).to(self.device)
        if monte_carlo_dropout:
            self.generator.train()
        else:
            self.generator.eval()

        self.aux_data_keys = ["attention_mask"]
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
            def decoding_fn(input_ids, **decode_kwargs):
                return [self.tokenizer.decode(curr_id, **decode_kwargs).strip() for curr_id in input_ids]
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
                    allow_truncation: Optional[bool] = True) -> Dict:
        truncation_strategy = "longest_first" if allow_truncation else "do_not_truncate"
        return self.encode_aligned(text_data,
                                   is_split_into_units=is_split_into_units,
                                   truncation_strategy=truncation_strategy)

    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor,
                 num_samples: int, **aux_data) -> Dict:
        return TransformersMLMGenerationMixin.generate(self, input_ids=input_ids, perturbable_mask=perturbable_mask,
                                                       num_samples=num_samples, **aux_data)

    def generate_masked_samples(self, input_ids: torch.Tensor, generation_mask: torch.Tensor, **generation_kwargs):
        return TransformersMLMGenerationMixin.generate_masked_samples(self, input_ids=input_ids,
                                                                      generation_mask=generation_mask,
                                                                      **generation_kwargs)


class XLMRobertaForMaskedLMGenerator(RobertaForMaskedLMGenerator, TransformersAlignedTokenizationMixin):
    def __init__(self, tokenizer_name, model_name, max_seq_len, batch_size=8, device="cuda",
                 strategy="top_k", top_p=0.9, top_k=5, monte_carlo_dropout: Optional[bool] = False):
        SampleGenerator.__init__(self, max_seq_len=max_seq_len, batch_size=batch_size, device=device,
                                 strategy=strategy, top_p=top_p, top_k=top_k)
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name

        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(self.tokenizer_name, add_prefix_space=True)
        self.generator = XLMRobertaForMaskedLM.from_pretrained(self.model_name, return_dict=True).to(self.device)
        if monte_carlo_dropout:
            self.generator.train()
        else:
            self.generator.eval()

        self.aux_data_keys = ["attention_mask"]
        self.special_tokens_set = set(self.tokenizer.all_special_ids)


class XLMRobertaForControlledMaskedLMGenerator(XLMRobertaForMaskedLMGenerator, TransformersCMLMGenerationMixin,
                                               TransformersAlignedTokenizationMixin):
    def __init__(self, tokenizer_name, model_name, control_labels: List[str], max_seq_len, batch_size=8, device="cuda",
                 strategy="top_k", top_p=0.9, top_k=5, label_weights: Optional[List] = None,
                 monte_carlo_dropout: Optional[bool] = False):
        super().__init__(tokenizer_name=tokenizer_name, model_name=model_name, max_seq_len=max_seq_len,
                         batch_size=batch_size, device=device, strategy=strategy, top_p=top_p, top_k=top_k,
                         monte_carlo_dropout=monte_carlo_dropout)

        assert all(curr_control in self.tokenizer.all_special_tokens for curr_control in control_labels)
        self.control_labels = torch.tensor(self.tokenizer.encode(control_labels, add_special_tokens=False,
                                                                 is_split_into_words=True))
        self.control_labels_str = control_labels

        # make it impossible to sample control labels (those are set in place)
        def mask_control(logits, **kwargs):
            logits[:, self.control_labels] = -float("inf")
            return logits
        self.filters = [mask_control] + self.filters

        self.label_weights = label_weights
        if self.label_weights is None:
            self.label_weights = [1.0] * len(self.control_labels)
        self.label_weights = torch.tensor(self.label_weights)
        self.label_weights /= torch.sum(self.label_weights)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor, num_samples: Optional[int],
                 **generation_kwargs):
        return TransformersCMLMGenerationMixin.generate(self, input_ids=input_ids, perturbable_mask=perturbable_mask,
                                                        num_samples=num_samples,
                                                        **generation_kwargs)

    @torch.no_grad()
    def generate_masked_samples(self, input_ids: torch.Tensor, generation_mask: torch.Tensor,
                                **generation_kwargs):
        return TransformersCMLMGenerationMixin.generate_masked_samples(self, input_ids=input_ids,
                                                                       generation_mask=generation_mask,
                                                                       **generation_kwargs)


if __name__ == "__main__":
    NUM_SAMPLES = 10
    GENERATOR_HANDLE = "/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/language_models/xlm-roberta-base-xnli-cmlm"
    generator = XLMRobertaForControlledMaskedLMGenerator(tokenizer_name=GENERATOR_HANDLE, model_name=GENERATOR_HANDLE,
                                                         batch_size=4, max_seq_len=42,
                                                         device="cpu", strategy="top_p", top_p=0.0001,
                                                         monte_carlo_dropout=True,
                                                         control_labels=["<ENTAILMENT>", "<NEUTRAL>", "<CONTRADICTION>"])

    ex = ("A shirtless man skateboards on a ledge", "A man without a shirt")
    pretokenized_ex = (ex[0].split(" "), ex[1].split(" "))
    encoded = generator.to_internal([pretokenized_ex], is_split_into_units=True)
    mask = torch.logical_and(torch.randint(2, (NUM_SAMPLES, encoded["input_ids"].shape[1])),
                             encoded["perturbable_mask"].repeat((NUM_SAMPLES, 1)))
    # generated = generator.generate_masked_samples(encoded["input_ids"],
    #                                               generation_mask=mask,
    #                                               control_labels=["<CONTRADICTION>" for _ in range(NUM_SAMPLES)],
    #                                               **encoded["aux_data"])
    generated = generator.generate(encoded["input_ids"], encoded["perturbable_mask"], num_samples=NUM_SAMPLES,
                                   control_labels=["<NEUTRAL>" for _ in range(NUM_SAMPLES)],
                                   **encoded["aux_data"])["input_ids"]

    for curr_ex in generator.from_internal(generated, skip_special_tokens=False, **encoded["aux_data"]):
        print(curr_ex)
