import json
import os
import pickle
import warnings
from collections import Counter
from typing import Tuple, Union, Dict, List, Optional, Mapping

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

from explain_nlp.methods.decoding import greedy_decoding, top_p_decoding, top_p_filtering, top_k_filtering
from explain_nlp.methods.utils import extend_tensor


class SampleGenerator:
    def mask_token(self):
        raise NotImplementedError

    def mask_token_id(self):
        raise NotImplementedError

    def convert_ids_to_tokens(self, ids: torch.Tensor) -> List[List[str]]:
        """ Convert integer-encoded tokens to str-encoded tokens, but keep them split."""
        raise NotImplementedError

    def from_internal(self, encoded_data: torch.Tensor, skip_special_tokens=True) -> List[Union[str, Tuple[str, ...]]]:
        """ Convert from internal generator representation to text."""
        raise NotImplementedError

    def to_internal(self, text_data: List[Union[str, Tuple[str, ...]]]) -> Dict:
        """ Convert from text to internal generator representation.
        Make sure to include 'input_ids', 'perturbable_mask' and 'aux_data' in the returned dictionary."""
        raise NotImplementedError

    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor, num_samples: int,
                 label: int, **aux_data) -> Dict:
        raise NotImplementedError


class GPTLMGenerator(SampleGenerator):
    def __init__(self, tokenizer_name, model_name, batch_size=2, max_seq_len=42, device="cuda",
                 top_p: Optional[float] = None, top_k: Optional[int] = 5, threshold: Optional[float] = 0.1):
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.top_p = top_p
        self.top_k = top_k
        self.threshold = threshold

        assert device in ["cpu", "cuda"]
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Device is set to 'cuda', but no CUDA device could be found. If you want to run the model "
                             "on CPU, set device to 'cpu'")
        self.device = torch.device(device)

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

    def from_internal(self, encoded_data, skip_special_tokens=True):
        decoded_data = []
        for idx_example in range(len(encoded_data)):
            curr_example = encoded_data[idx_example]
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


class GPTControlledLMGenerator(GPTLMGenerator):
    def __init__(self, tokenizer_name, model_name, control_labels: List[str], batch_size=2, max_seq_len=42, device="cuda",
                 strategy: Optional[str] = "greedy", top_p: Optional[float] = None, top_k: Optional[int] = 5,
                 threshold: Optional[float] = 0.1, label_weights: List = None,
                 generate_expected_examples: Optional[bool] = False):
        super().__init__(tokenizer_name=tokenizer_name, model_name=model_name, batch_size=batch_size,
                         max_seq_len=max_seq_len, device=device, top_p=top_p, top_k=top_k, threshold=threshold)

        assert all(curr_control in self.tokenizer.all_special_tokens for curr_control in control_labels)
        self.control_labels = torch.tensor(self.tokenizer.encode(control_labels, add_special_tokens=False))
        self.control_labels_str = control_labels

        self.label_weights = label_weights
        if self.label_weights is None:
            self.label_weights = [1.0] * len(self.control_labels)
        self.label_weights = torch.tensor(self.label_weights)

        # TODO: beam search = "global"/"sequence_level"?
        self.strategy_type = "token_level"
        if strategy == "top_p":
            assert self.top_p is not None
            self.filtering_strategy = lambda logits: top_p_filtering(logits, top_p=self.top_p)
        elif strategy == "top_k":
            assert self.top_k is not None
            self.filtering_strategy = lambda logits: top_k_filtering(logits, top_k=self.top_k)
        else:
            raise NotImplementedError(f"Unsupported filtering strategy: '{strategy}'")

        #  Denotes whether to change control label at every step in order to try generating "expected example"
        self.generate_expected = generate_expected_examples
        if self.generate_expected:  # TODO
            raise NotImplementedError("'generate_expected_examples' is currently unimplemented for GPT "
                                      "controlled language modeling")

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
                    probas = F.softmax(logits, dim=-1)

                    preds = torch.multinomial(probas, num_samples=1)
                    curr_generated_ids[generation_mask, idx_step] = preds[:, 0].cpu()

                is_eos = torch.logical_or(is_eos, curr_generated_ids[:, idx_step] == self.tokenizer.eos_token_id)

            generated_input_ids[s_b: e_b] = curr_generated_ids

        generated_input_ids[:, 0] = self.tokenizer.bos_token_id
        generated_input_ids[:, -1] = self.tokenizer.eos_token_id

        return {
            "input_ids": generated_input_ids
        }


class BertForMaskedLMGenerator(SampleGenerator):
    MLM_MAX_MASK_PROPORTION = 0.15

    def __init__(self, tokenizer_name, model_name, batch_size=8, max_seq_len=64, device="cuda",
                 strategy: Optional[str] = "top_k", top_p: Optional[float] = None, top_k: Optional[int] = 5,
                 threshold: Optional[float] = 0.1, generate_cover: Optional[bool] = False):
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.generate_cover = generate_cover
        self.strategy = strategy
        assert not (self.generate_cover and self.strategy == "greedy")
        self.top_p = top_p
        self.top_k = top_k
        self.threshold = threshold

        self.filtering_strategy = None
        if strategy == "top_p":
            assert self.top_p is not None
            self.filtering_strategy = lambda logits: top_p_filtering(logits, top_p=self.top_p)
        elif strategy == "top_k":
            assert self.top_k is not None
            self.filtering_strategy = lambda logits: top_k_filtering(logits, top_k=self.top_k)
        else:
            raise NotImplementedError(f"Unsupported filtering strategy: '{strategy}'")

        assert self.batch_size > 1 and self.batch_size % 2 == 0
        assert device in ["cpu", "cuda"]
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Device is set to 'cuda', but no CUDA device could be found. If you want to run the model "
                             "on CPU, set device to 'cpu'")
        self.device = torch.device(device)

        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        self.generator = BertForMaskedLM.from_pretrained(self.model_name, return_dict=True).to(self.device)
        self.generator.eval()

    @property
    def mask_token(self) -> str:
        return self.tokenizer.mask_token

    @property
    def mask_token_id(self) -> int:
        return self.tokenizer.mask_token_id

    def convert_ids_to_tokens(self, ids):
        return [self.tokenizer.convert_ids_to_tokens(curr_ids) for curr_ids in ids.tolist()]

    def from_internal(self, encoded_data, skip_special_tokens=True):
        decoded_data = []
        for idx_example in range(encoded_data.shape[0]):
            curr_example = encoded_data[idx_example]
            sep_tokens = torch.nonzero(curr_example == self.tokenizer.sep_token_id, as_tuple=False)
            end = int(sep_tokens[-1])

            # Multiple sequences present: [CLS] <seq1> [SEP] <seq2> [SEP] -> (<seq1>, <seq2>)
            if sep_tokens.shape[0] == 2:
                bnd = int(sep_tokens[0])
                seq1 = self.tokenizer.decode(encoded_data[idx_example, 1: bnd],
                                             skip_special_tokens=skip_special_tokens)
                seq2 = self.tokenizer.decode(encoded_data[idx_example, bnd + 1: end],
                                             skip_special_tokens=skip_special_tokens)
                decoded_data.append((seq1, seq2))
            else:
                decoded_data.append(self.tokenizer.decode(encoded_data[idx_example, :end],
                                                          skip_special_tokens=skip_special_tokens))

        return decoded_data

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
    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor,
                 num_samples, label=None, **aux_data) -> Dict:
        raise NotImplementedError("Needs to be reimplemented the new way (as does GPTLM)")


class BertForControlledMaskedLMGenerator(BertForMaskedLMGenerator):
    def __init__(self, tokenizer_name, model_name, control_labels: List[str], batch_size=8, max_seq_len=64, device="cuda",
                 strategy: Optional[str] = "greedy", top_p: Optional[float] = None, top_k: Optional[int] = 5,
                 threshold: Optional[float] = 0.1, label_weights: List = None, unique_dropout: Optional[float] = 0.0,
                 generate_expected_examples: Optional[bool] = False, generate_cover: Optional[bool] = False):
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

        # TODO: beam search = "global"/"sequence_level"?
        self.strategy_type = "token_level"

        if generate_expected_examples:
            warnings.warn("'generate_expected_examples' is deprecated for BERT controlled MLM")

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


# Just so that the tokenizer class matches the model class
TrigramTokenizer = BertTokenizer


class TrigramMLM:
    def __init__(self,
                 vocab_size,
                 unk_token_id,
                 frequencies: Optional[Mapping] = None,
                 norm_consts: Optional[Mapping] = None,
                 unigram_frequencies: Optional[Mapping] = None):
        self.unk_token_id = unk_token_id
        self.vocab_size = vocab_size

        # {(left, middle, right) -> count}
        self.frequencies = Counter() if frequencies is None else frequencies
        # {(left, right) -> sum of counts over all middle, given left, right}
        self.norm_consts = Counter() if norm_consts is None else norm_consts

        # {token -> count}
        self.unigram_frequencies = Counter() if unigram_frequencies is None else unigram_frequencies
        self.unigram_count = sum(self.unigram_frequencies.values())
        self.unigram_distribution = {"i": [], "proba": []}

        self.distributions = {}
        self._calculate_distribution()

    def _calculate_distribution(self):
        self.distributions = {(left, right): {"i": [], "proba": []} for left, _, right in self.frequencies.keys()}
        for (left, mid, right), count in self.frequencies.items():
            existing_i = self.distributions[(left, right)]["i"]
            existing_proba = self.distributions[(left, right)]["proba"]

            existing_i.append(mid)
            existing_proba.append(count / self.norm_consts[(left, right)])

        self.unigram_count = sum(self.unigram_frequencies.values())
        for token, count in self.unigram_frequencies.items():
            self.unigram_distribution["i"].append(token)
            self.unigram_distribution["proba"].append(count / self.unigram_count)

    def train(self, input_ids, special_tokens_mask):
        for curr_input, curr_special in tqdm(zip(input_ids, special_tokens_mask), total=len(input_ids)):
            for left, left_special, mid, mid_special, right, right_special in zip(curr_input, curr_special,
                                                                                  curr_input[1:], curr_special[1:],
                                                                                  curr_input[2:], curr_special[2:]):
                # don't predict special tokens, can condition on them though
                if mid_special or mid == self.unk_token_id:
                    continue

                self.frequencies.update({(left, mid, right): 1})
                self.norm_consts.update({(left, right): 1})

            for token, is_special in zip(curr_input, curr_special):
                if not is_special:
                    self.unigram_frequencies.update({token: 1})

        self._calculate_distribution()

    def _predict_with_unk_backoff(self, left, right):
        _left, _right = left.item(), right.item()
        pred_distr = self.distributions.get((_left, _right), None)
        if pred_distr is not None:
            return torch.tensor(pred_distr["i"]), torch.tensor(pred_distr["proba"])

        # backoff: prev token = UNK
        pred_distr = self.distributions.get((self.unk_token_id, _right), None)
        if pred_distr is not None:
            return torch.tensor(pred_distr["i"]), torch.tensor(pred_distr["proba"])

        # backoff: next token = UNK
        pred_distr = self.distributions.get((_left, self.unk_token_id), None)
        if pred_distr is not None:
            return torch.tensor(pred_distr["i"]), torch.tensor(pred_distr["proba"])

        # backoff: prev = next = UNK
        pred_distr = self.distributions.get((self.unk_token_id, self.unk_token_id), None)
        if pred_distr is not None:
            return torch.tensor(pred_distr["i"]), torch.tensor(pred_distr["proba"])

        return torch.tensor(self.unigram_distribution["i"]), torch.tensor(self.unigram_distribution["proba"])

    def __call__(self, input_ids, **kwargs):
        num_examples, max_seq_len = input_ids.shape
        logits = torch.zeros((num_examples, max_seq_len, self.vocab_size), dtype=torch.float32)
        logits[:, :, :] = -float("inf")

        for idx_example in range(num_examples):
            for idx_token in range(max_seq_len):
                original_token = input_ids[idx_example, idx_token]

                # No left or right side to condition on, predict ground truth with 100% certainty
                if idx_token == 0 or idx_token == max_seq_len - 1:
                    logits[idx_example, idx_token, original_token] = torch.log(torch.tensor(1.0))
                    continue

                prev_token = input_ids[idx_example, idx_token - 1]
                next_token = input_ids[idx_example, idx_token + 1]
                tokens, probas = self._predict_with_unk_backoff(prev_token, next_token)
                logits[idx_example, idx_token, tokens] = torch.log(probas)

        return {
            "logits": logits
        }

    @staticmethod
    def from_pretrained(pretrained_path):
        if not os.path.exists(pretrained_path):
            raise ValueError(f"Directory '{pretrained_path}' does not exist")

        with open(os.path.join(pretrained_path, "trigram_config.json"), "r") as f:
            config = json.load(f)

        with open(os.path.join(pretrained_path, "frequencies.pkl"), "rb") as f:
            frequencies = pickle.load(f)

        with open(os.path.join(pretrained_path, "norm_consts.pkl"), "rb") as f:
            norm_consts = pickle.load(f)

        with open(os.path.join(pretrained_path, "unigram_frequencies.pkl"), "rb") as f:
            unigram_frequencies = pickle.load(f)

        return TrigramMLM(vocab_size=config["vocab_size"],
                          unk_token_id=config["unk_token_id"],
                          frequencies=frequencies,
                          norm_consts=norm_consts,
                          unigram_frequencies=unigram_frequencies)

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        with open(os.path.join(save_directory, "trigram_config.json"), "w") as f:
            json.dump({
                "unk_token_id": self.unk_token_id,
                "vocab_size": self.vocab_size
            }, fp=f, indent=4)

        with open(os.path.join(save_directory, "frequencies.pkl"), "wb") as f:
            pickle.dump(self.frequencies, file=f)

        with open(os.path.join(save_directory, "norm_consts.pkl"), "wb") as f:
            pickle.dump(self.norm_consts, file=f)

        with open(os.path.join(save_directory, "unigram_frequencies.pkl"), "wb") as f:
            pickle.dump(self.unigram_frequencies, file=f)


class TrigramForMaskedLMGenerator(SampleGenerator):
    def __init__(self, tokenizer_name, model_name, batch_size=8, max_seq_len=64, device="cuda",
                 strategy: Optional[str] = "top_k", top_p: Optional[float] = None, top_k: Optional[int] = 5,
                 threshold: Optional[float] = 0.1, generate_cover: Optional[bool] = False):
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.generate_cover = generate_cover
        self.strategy = strategy
        assert not (self.generate_cover and self.strategy == "greedy")
        self.top_p = top_p
        self.top_k = top_k
        self.threshold = threshold

        self.filtering_strategy = None
        if strategy == "top_p":
            assert self.top_p is not None
            self.filtering_strategy = lambda logits: top_p_filtering(logits, top_p=self.top_p)
        elif strategy == "top_k":
            assert self.top_k is not None
            self.filtering_strategy = lambda logits: top_k_filtering(logits, top_k=self.top_k)
        else:
            raise NotImplementedError(f"Unsupported filtering strategy: '{strategy}'")

        assert self.batch_size > 1 and self.batch_size % 2 == 0
        assert device in ["cpu", "cuda"]
        if device == "cuda":
            warnings.warn(f"Device 'cuda' is not implemented for trigram MLM, defaulting to 'cpu'")
        self.device = torch.device(device)

        self.tokenizer = TrigramTokenizer.from_pretrained(self.tokenizer_name)
        self.generator = TrigramMLM.from_pretrained(self.model_name)

    @property
    def mask_token(self) -> str:
        return self.tokenizer.mask_token

    @property
    def mask_token_id(self) -> int:
        return self.tokenizer.mask_token_id

    def convert_ids_to_tokens(self, ids):
        return [self.tokenizer.convert_ids_to_tokens(curr_ids) for curr_ids in ids.tolist()]

    def from_internal(self, encoded_data, skip_special_tokens=True):
        decoded_data = []
        for idx_example in range(encoded_data.shape[0]):
            sep_tokens = torch.nonzero(encoded_data[idx_example] == self.tokenizer.sep_token_id, as_tuple=False)
            end = int(sep_tokens[-1])

            # Multiple sequences present: [CLS] <seq1> [SEP] <seq2> [SEP] -> (<seq1>, <seq2>)
            if sep_tokens.shape[0] == 2:
                bnd = int(sep_tokens[0])
                seq1 = self.tokenizer.decode(encoded_data[idx_example, 1 :bnd],
                                             skip_special_tokens=skip_special_tokens)
                seq2 = self.tokenizer.decode(encoded_data[idx_example, bnd + 1: end],
                                             skip_special_tokens=skip_special_tokens)
                decoded_data.append((seq1, seq2))
            else:
                decoded_data.append(self.tokenizer.decode(encoded_data[idx_example, :end],
                                                          skip_special_tokens=skip_special_tokens))

        return decoded_data

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
    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor,
                 num_samples, label=None, **aux_data) -> Dict:
        raise NotImplementedError("Needs to be reimplemented the new way")


if __name__ == "__main__":
    # generator = BertForControlledMaskedLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert_snli_clm_best",
    #                                                model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert_snli_clm_best",
    #                                                control_labels=["<ENTAILMENT>", "<NEUTRAL>", "<CONTRADICTION>"],
    #                                                batch_size=2,
    #                                                device="cpu",
    #                                                strategy="top_k",
    #                                                top_k=3,
    #                                                generate_cover=True)

    generator = GPTControlledLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/gpt_snli_clm_maxseqlen42",
                                         model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/gpt_snli_clm_maxseqlen42",
                                         control_labels=["<ENTAILMENT>", "<NEUTRAL>", "<CONTRADICTION>"],
                                         batch_size=2,
                                         device="cpu",
                                         strategy="top_p",
                                         top_p=0.9)

    # generator = GPTLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/gpt_snli_lm_maxseqlen42",
    #                            model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/gpt_snli_lm_maxseqlen42",
    #                            batch_size=2,
    #                            max_seq_len=42,
    #                            device="cpu")

    seq = ("A patient is being worked on by doctors and nurses", "A man is sleeping.")
    label = 0  # "entailment"
    encoded = generator.to_internal([seq])

    generated = generator.generate(encoded["input_ids"], label=label, perturbable_mask=encoded["perturbable_mask"],
                                   num_samples=10, **encoded["aux_data"])["input_ids"]

    for curr_ids in generated:
        print(generator.tokenizer.decode(curr_ids, skip_special_tokens=False))

