import warnings
from typing import Optional, List, Union, Tuple, Dict

import torch
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, BertTokenizer, BertForMaskedLM, RobertaTokenizer, \
    RobertaForMaskedLM

from explain_nlp.generation.generation_base import SampleGenerator
from explain_nlp.methods.decoding import greedy_decoding, top_p_decoding
from explain_nlp.methods.utils import extend_tensor


class GPTLMGenerator(SampleGenerator):
    def __init__(self, tokenizer_name, model_name, max_seq_len, batch_size=2, device="cuda",
                 strategy="top_p", top_p=0.9, top_k=5, threshold=0.1):
        super().__init__(max_seq_len=max_seq_len, batch_size=batch_size, device=device,
                         strategy=strategy, top_p=top_p, top_k=top_k, threshold=threshold)

        self.tokenizer_name = tokenizer_name
        self.model_name = model_name

        self.tokenizer = OpenAIGPTTokenizer.from_pretrained(self.tokenizer_name)
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

    @property
    def mask_token(self):
        return self.tokenizer.mask_token

    @property
    def mask_token_id(self):
        return self.tokenizer.mask_token_id

    def convert_ids_to_tokens(self, ids):
        return [self.tokenizer.convert_ids_to_tokens(curr_ids) for curr_ids in ids.tolist()]

    def from_internal(self, encoded_data, skip_special_tokens: bool = True, take_as_single_sequence: bool = False,
                      **kwargs):
        decoded_data = []
        for idx_example in range(len(encoded_data)):
            curr_example = encoded_data[idx_example]
            if take_as_single_sequence:
                decoded_data.append(self.tokenizer.decode(encoded_data[idx_example], skip_special_tokens=skip_special_tokens))
            else:
                sep_tokens = torch.nonzero(curr_example == self.tokenizer.sep_token_id, as_tuple=False)
                eos_tokens = torch.nonzero(curr_example == self.tokenizer.eos_token_id, as_tuple=False)
                end = int(eos_tokens[-1])

                # Multiple sequences present: <BOS> <seq1> <SEP> <seq2> <EOS> -> (<seq1>, <seq2>)
                if sep_tokens.shape[0] == 1:
                    bnd = int(sep_tokens[0])
                    seq1 = self.tokenizer.decode(curr_example[1: bnd], skip_special_tokens=skip_special_tokens)
                    seq2 = self.tokenizer.decode(curr_example[bnd + 1: end], skip_special_tokens=skip_special_tokens)
                    decoded_data.append((seq1, seq2))
                else:
                    decoded_data.append(self.tokenizer.decode(curr_example[: end], skip_special_tokens=skip_special_tokens))

        return decoded_data

    def to_internal(self, text_data: List[Union[str, Tuple[str, ...]]]) -> Dict:
        _text_data = []
        for curr_text in text_data:
            if isinstance(curr_text, str):
                _text_data.append(
                    f"{self.tokenizer.bos_token} {curr_text[0]} {self.tokenizer.sep_token}")
            else:  # tuple/list
                _text_data.append(
                    f"{self.tokenizer.bos_token} {curr_text[0]} {self.tokenizer.sep_token} {curr_text[1]} {self.tokenizer.eos_token}")

        res = self.tokenizer.batch_encode_plus(_text_data, return_special_tokens_mask=True, return_tensors="pt",
                                               padding="max_length", max_length=self.max_seq_len,
                                               truncation="longest_first")

        fixed_perturbable_mask = []
        for idx_example in range(len(text_data)):
            sequence = res["input_ids"][idx_example].tolist()
            mask = res["special_tokens_mask"][idx_example].tolist()
            fixed_perturbable = [not (is_special | (enc_token in self.tokenizer.all_special_ids))
                                 for is_special, enc_token in zip(mask, sequence)]
            fixed_perturbable_mask.append(fixed_perturbable)

        formatted_res = {
            "input_ids": res["input_ids"],
            "perturbable_mask": torch.tensor(fixed_perturbable_mask),
            "aux_data": {
                "attention_mask": res["attention_mask"]
            }
        }

        return formatted_res

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

                    curr_logits = self.filtering_strategy(curr_logits)
                    probas = torch.softmax(curr_logits, dim=-1)
                    preds = torch.multinomial(probas, num_samples=1)[:, 0].cpu()

                    curr_inputs[is_feature_masked[:, pos], s_c + pos] = preds[is_feature_masked[:, pos]]

        return eff_input_ids


class GPTControlledLMGenerator(GPTLMGenerator):
    def __init__(self, tokenizer_name, model_name, control_labels: List[str], max_seq_len,
                 batch_size=2, device="cuda", strategy="greedy", top_p=0.9, top_k=5, threshold=0.1,
                 label_weights: Optional[List] = None):
        super().__init__(tokenizer_name=tokenizer_name, model_name=model_name, batch_size=batch_size,
                         max_seq_len=max_seq_len, device=device, top_p=top_p, top_k=top_k, threshold=threshold,
                         strategy=strategy)

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


class BertForMaskedLMGenerator(SampleGenerator):
    def __init__(self, tokenizer_name, model_name, max_seq_len, batch_size=8, device="cuda",
                 strategy="top_k", top_p=0.9, top_k=5, threshold=0.1,
                 generate_cover: Optional[bool] = False, monte_carlo_dropout: Optional[bool] = False,
                 allowed_values: Optional[List[torch.Tensor]] = None):
        super().__init__(max_seq_len=max_seq_len, batch_size=batch_size, device=device,
                         strategy=strategy, top_p=top_p, top_k=top_k, threshold=threshold)
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.generate_cover = generate_cover

        assert self.batch_size > 1 and self.batch_size % 2 == 0

        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        self.generator = BertForMaskedLM.from_pretrained(self.model_name, return_dict=True).to(self.device)
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

    def convert_ids_to_tokens(self, ids):
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

    def to_internal(self, text_data):
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

        # Note: currently assuming generation additional data is same for all samples
        eff_aux_data = {k: generation_kwargs[k].repeat((self.batch_size, 1)).to(self.device)
                        for k in ["token_type_ids", "attention_mask"]}

        mask_size = 1
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        num_total_chunks = (num_features + mask_size - 1) // mask_size

        for idx_batch in range(num_batches):
            s_b, e_b = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size

            curr_inputs = eff_input_ids[s_b: e_b]
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
                curr_aux_data = {k: v[: curr_batch_size] for k, v in eff_aux_data.items()}

                logits = self.generator(input_ids=curr_inputs.to(self.device), **curr_aux_data)["logits"]
                for pos in range(curr_mask_size):
                    curr_logits = self.mask_impossible(logits[:, s_c + pos, :], position=(s_c + pos))
                    curr_logits = self.filtering_strategy(curr_logits)

                    probas = torch.softmax(curr_logits, dim=-1)
                    preds = torch.multinomial(probas, num_samples=1)[:, 0].cpu()

                    curr_inputs[is_feature_masked[:, pos], s_c + pos] = preds[is_feature_masked[:, pos]]

        return eff_input_ids


class BertForControlledMaskedLMGenerator(BertForMaskedLMGenerator):
    def __init__(self, tokenizer_name, model_name, control_labels: List[str], max_seq_len,
                 batch_size=8, device="cuda", strategy="greedy", top_p=0.9, top_k=5, threshold=0.1,
                 label_weights: Optional[List] = None, unique_dropout: Optional[float] = 0.0,
                 generate_cover: Optional[bool] = False):
        super().__init__(tokenizer_name=tokenizer_name, model_name=model_name,
                         batch_size=batch_size, max_seq_len=max_seq_len, device=device,
                         strategy=strategy, top_p=top_p, top_k=top_k, threshold=threshold,
                         generate_cover=generate_cover)
        self.unique_dropout = unique_dropout

        assert all(curr_control in self.tokenizer.all_special_tokens for curr_control in control_labels)
        self.control_labels = torch.tensor(self.tokenizer.encode(control_labels, add_special_tokens=False))
        self.control_labels_str = control_labels

        self.label_weights = label_weights
        if self.label_weights is None:
            self.label_weights = [1.0] * len(self.control_labels)
        self.label_weights = torch.tensor(self.label_weights)
        self.label_weights /= torch.sum(self.label_weights)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor,
                 num_samples: Optional[int], **aux_data):
        if self.generate_cover:
            print("Warning: generate_cover=True, meaning the 'num_samples' argument is ignored in favour of number "
                  "of samples, determined by cover")

        # Make room for control label at start of sequence (at pos. 1)
        extended_input_ids = extend_tensor(input_ids).repeat((num_samples, 1))
        extended_pert_mask = extend_tensor(perturbable_mask)
        extended_token_type_ids = extend_tensor(aux_data["token_type_ids"]).repeat((self.batch_size, 1)).to(self.device)
        extended_attention_mask = extend_tensor(aux_data["attention_mask"]).repeat((self.batch_size, 1)).to(self.device)

        # Fairly allocate generated samples according to label distribution
        selected_control_labels = []
        for idx_label in range(self.control_labels.shape[0]):
            allocated_samples = int(torch.floor(self.label_weights[idx_label] * num_samples))
            selected_control_labels.extend([self.control_labels[idx_label].item()] * allocated_samples)

        # In case the samples can't be allocated perfectly (due to rounding), randomly allocate them according to dist.
        if len(selected_control_labels) != num_samples:
            leftover_control_labels = torch.multinomial(self.label_weights,
                                                        num_samples=(num_samples - len(selected_control_labels)),
                                                        replacement=False)
            selected_control_labels.extend(self.control_labels[leftover_control_labels].tolist())

        selected_control_labels = torch.tensor(selected_control_labels)
        extended_input_ids[:, 1] = selected_control_labels
        extended_attention_mask[:, 1] = 1
        perturbable_inds = torch.arange(extended_input_ids.shape[1])[extended_pert_mask[0]]

        if self.strategy == "greedy":
            # If generation order was not shuffled, greedy decoding would produce identical samples
            weights = torch.zeros_like(extended_input_ids, dtype=torch.float32)
            weights[:, perturbable_inds] = 1
            generation_order = torch.multinomial(weights, num_samples=perturbable_inds.shape[0], replacement=False)
        else:
            generation_order = perturbable_inds.repeat((num_samples, 1))

        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for idx_batch in range(num_batches):
            s_b, e_b = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
            curr_input_ids = extended_input_ids[s_b: e_b]  # view
            curr_gen_order = generation_order[s_b: e_b]

            curr_batch_size = curr_input_ids.shape[0]
            batch_indexer = torch.arange(curr_batch_size)

            for i in range(curr_gen_order.shape[1]):
                curr_indices = curr_gen_order[:, i]
                curr_input_ids[batch_indexer, curr_indices] = self.mask_token_id

                res = self.generator(input_ids=curr_input_ids.to(self.device),
                                     token_type_ids=extended_token_type_ids[:curr_batch_size],
                                     attention_mask=extended_attention_mask[:curr_batch_size])

                logits = res["logits"]  # [batch_size, max_seq_len, |V|]

                curr_masked_logits = logits[batch_indexer, curr_indices, :]
                curr_masked_logits = self.filtering_strategy(curr_masked_logits)

                curr_probas = torch.softmax(curr_masked_logits, dim=-1)
                preds = torch.multinomial(curr_probas, num_samples=1)
                curr_input_ids[batch_indexer, curr_indices] = preds[:, 0].cpu()

        valid_tokens = torch.ones_like(extended_pert_mask)
        valid_tokens[0, 1] = False

        return {
            "input_ids": extended_input_ids[:, valid_tokens[0]]
        }

    @torch.no_grad()
    def generate_masked_samples(self, input_ids: torch.Tensor,
                                generation_mask: torch.Tensor,
                                **generation_kwargs):
        eff_input_ids = extend_tensor(input_ids)
        eff_generation_mask = extend_tensor(generation_mask)

        # Note: currently assuming generation additional data is same for all samples
        eff_aux_data = {k: generation_kwargs[k].repeat((self.batch_size, 1)).to(self.device)
                        for k in ["token_type_ids", "attention_mask"]}
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


class RobertaForMaskedLMGenerator(SampleGenerator):
    def __init__(self, tokenizer_name, model_name, max_seq_len, batch_size=8, device="cuda",
                 strategy="top_k", top_p=0.9, top_k=5, threshold=0.1, monte_carlo_dropout: Optional[bool] = False,
                 allowed_values: Optional[List[torch.Tensor]] = None):
        super().__init__(max_seq_len=max_seq_len, batch_size=batch_size, device=device,
                         strategy=strategy, top_p=top_p, top_k=top_k, threshold=threshold)
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name

        self.tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_name)
        self.generator = RobertaForMaskedLM.from_pretrained(self.model_name, return_dict=True).to(self.device)
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

        decoded_data = []
        for idx_example in range(num_ex):
            if take_as_single_sequence:
                decoded_data.append(self.tokenizer.decode(encoded_data[idx_example], skip_special_tokens=skip_special_tokens))
            else:
                curr_attendable = attention_fn(idx_example).bool()
                curr_input_ids = encoded_data[idx_example][curr_attendable]
                sep_positions = torch.flatten(torch.nonzero(curr_input_ids == self.tokenizer.sep_token_id,
                                                            as_tuple=False))

                if sep_positions.shape[0] == 1:
                    # <s> <seq> </s>
                    decoded_data.append(self.tokenizer.decode(curr_input_ids[1: -1],
                                                              skip_special_tokens=skip_special_tokens))
                else:
                    # <s> <seq1> </s></s> <seq2> </s>
                    starts = [1] + (sep_positions[1::2] + 1).tolist()
                    ends = sep_positions[::2].tolist() + [-1]

                    multiple_sequences = []
                    for s, e in zip(starts, ends):
                        multiple_sequences.append(self.tokenizer.decode(curr_input_ids[s: e],
                                                                        skip_special_tokens=skip_special_tokens))
                    decoded_data.append(tuple(multiple_sequences))

        return decoded_data

    def from_internal_precise(self, encoded_data, skip_special_tokens=True):
        # TODO: properly set `is_continuation` (based on Ġ and add_prefix_space option of tokenizer)
        converted = {
            "decoded_data": [],
            "is_continuation": []
        }
        for idx_example in range(encoded_data.shape[0]):
            curr_example = encoded_data[idx_example]
            sep_tokens = torch.flatten(torch.nonzero(curr_example == self.tokenizer.sep_token_id, as_tuple=False))
            end = int(sep_tokens[-1])

            print(curr_example)

            processed_example, is_continuation = [], []
            for el in curr_example:
                is_continuation.append(False)

                if skip_special_tokens and el.item() in self.special_tokens_set:
                    processed_example.append("")
                    continue

                str_tok = self.tokenizer.convert_ids_to_tokens(el.item())
                str_tok = self.tokenizer.convert_tokens_to_string(str_tok).strip()
                processed_example.append(str_tok)

            if sep_tokens.shape[0] == 1:
                # <s> <seq> </s>
                converted["decoded_data"].append(processed_example[1: end])
                converted["is_continuation"].append(is_continuation[1: end])
            else:
                # <s> <seq1> </s></s> <seq2> </s> -> (<seq1>, <seq2>)
                bnd = int(sep_tokens[0])
                converted["decoded_data"].append((processed_example[1: bnd],
                                                    processed_example[bnd + 2: end]))
                converted["is_continuation"].append((is_continuation[1: bnd],
                                                     is_continuation[bnd + 2: end]))

        return converted

    def to_internal(self, text_data: List[Union[str, Tuple[str, ...]]]) -> Dict:
        res = self.tokenizer.batch_encode_plus(text_data, return_special_tokens_mask=True, return_tensors="pt",
                                               padding="max_length", max_length=self.max_seq_len,
                                               truncation="longest_first")
        formatted_res = {
            "input_ids": res["input_ids"],
            "perturbable_mask": torch.logical_not(res["special_tokens_mask"]),
            "aux_data": {
                "attention_mask": res["attention_mask"]
            }
        }

        return formatted_res

    def generate_masked_samples(self, input_ids: torch.Tensor, generation_mask: torch.Tensor, **generation_kwargs):
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
                curr_aux_data = {k: v[: curr_batch_size] for k, v in eff_aux_data.items()}

                logits = self.generator(input_ids=curr_inputs.to(self.device), **curr_aux_data)["logits"]
                for pos in range(curr_mask_size):
                    curr_logits = self.mask_impossible(logits[:, s_c + pos, :], position=(s_c + pos))
                    curr_logits = self.filtering_strategy(curr_logits)

                    probas = torch.softmax(curr_logits, dim=-1)
                    preds = torch.multinomial(probas, num_samples=1)[:, 0].cpu()

                    curr_inputs[is_feature_masked[:, pos], s_c + pos] = preds[is_feature_masked[:, pos]]

        return eff_input_ids


if __name__ == "__main__":
    NUM_SAMPLES = 10
    generator = RobertaForMaskedLMGenerator(tokenizer_name="roberta-base", model_name="roberta-base",
                                            batch_size=10, max_seq_len=41,
                                            device="cpu",
                                            strategy="top_p",
                                            top_p=0.99,
                                            monte_carlo_dropout=False)

    ex = ("A shirtless man skateboards on a ledge.", "A man without a shirt")
    encoded = generator.to_internal([ex])
    generated = generator.generate_masked_samples(encoded["input_ids"],
                                                  generation_mask=encoded["perturbable_mask"].repeat((NUM_SAMPLES, 1)),
                                                  **encoded["aux_data"])

    for curr_ids in generated:
        print(generator.tokenizer.decode(curr_ids, skip_special_tokens=False))
