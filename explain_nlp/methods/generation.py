from typing import Tuple, Union, Dict, List, Optional

import torch
from transformers import BertTokenizer, BertForMaskedLM, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

from explain_nlp.methods.decoding import greedy_decoding, top_p_decoding, top_p_filtering, top_k_decoding

# TODO: mask, mask_token_id properties
# TODO: mask token in generator is not necessarily same as mask token in model (e.g. <MASK> vs [MASK])
from explain_nlp.methods.utils import extend_tensor


class SampleGenerator:
    def from_internal(self, encoded_data: torch.Tensor) -> List[Union[str, Tuple[str, ...]]]:
        """ Convert from internal generator representation to text."""
        raise NotImplementedError

    def to_internal(self, text_data: List[Union[str, Tuple[str, ...]]]) -> Dict:
        """ Convert from text to internal generator representation.
        Make sure to include 'input_ids', 'perturbable_mask' and 'aux_data' in the returned dictionary."""
        raise NotImplementedError

    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor, num_samples: int,
                 label: int, **aux_data) -> Dict:
        raise NotImplementedError


class GPTControlledLMGenerator(SampleGenerator):
    def __init__(self, tokenizer_name, model_name, possible_labels, batch_size=2, max_seq_len=42, device="cuda",
                 top_p: Optional[float] = None):
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.top_p = top_p

        assert device in ["cpu", "cuda"]
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Device is set to 'cuda', but no CUDA device could be found. If you want to run the model "
                             "on CPU, set device to 'cpu'")
        self.device = torch.device(device)

        self.tokenizer = OpenAIGPTTokenizer.from_pretrained(self.tokenizer_name)
        self.generator = OpenAIGPTLMHeadModel.from_pretrained(self.model_name).to(self.device)
        self.generator.eval()

        self.possible_labels = possible_labels
        self.labels_map = {}
        for curr_lbl in possible_labels:
            encoded_lbl = self.tokenizer.encode(f"<{curr_lbl.upper()}>")
            assert len(encoded_lbl) == 1  # make sure the label is actually a special token and doesn't get broken up
            self.labels_map[curr_lbl] = encoded_lbl[0]

        # Required to build sequence: <LABEL> <seq1> [<SEP> <seq2>] <EOS>
        assert self.tokenizer.sep_token_id is not None
        assert self.tokenizer.eos_token_id is not None

    def from_internal(self, encoded_data: torch.Tensor) -> List[Union[str, Tuple[str, ...]]]:
        # NOTE: similar to regular GPT, but remove the <LABEL> at the beginning of sequences (used for controlled LM)
        decoded_data = []
        for idx_example in range(encoded_data.shape[0]):
            sep_tokens = torch.nonzero(encoded_data[idx_example] == self.tokenizer.sep_token_id, as_tuple=False)

            # Multiple sequences present: <BOS> <seq1> <SEP> <seq2> <EOS> -> (<seq1>, <seq2>)
            if sep_tokens.shape[0] == 1:
                bnd = int(sep_tokens[0])
                seq1 = self.tokenizer.decode(encoded_data[idx_example, 1: bnd], skip_special_tokens=True)
                seq2 = self.tokenizer.decode(encoded_data[idx_example, bnd + 1:], skip_special_tokens=True)
                decoded_data.append((seq1, seq2))
            else:
                decoded_data.append(self.tokenizer.decode(encoded_data[idx_example, 1:], skip_special_tokens=True))

        return decoded_data

    def to_internal(self, text_data: List[Union[str, Tuple[str, ...]]]) -> Dict:
        res = self.tokenizer.batch_encode_plus(text_data, return_special_tokens_mask=True, return_tensors="pt",
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
    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor, num_samples: int, label: int,
                 **aux_data) -> Dict:
        num_features = int(input_ids.shape[1])

        perturbable_inds = torch.arange(num_features)[perturbable_mask[0]]
        masked_samples = input_ids.repeat((num_samples, 1))

        if self.top_p is None:
            def decoding_strategy(logits, ensure_diff_from):
                return greedy_decoding(logits, ensure_diff_from)
        else:
            def decoding_strategy(logits, ensure_diff_from):
                return top_p_decoding(logits, self.top_p, ensure_diff_from)

        # Inject some randomness by generating first token based on randomly selected label
        rnd_labels = torch.randint(len(self.possible_labels), (num_samples,))
        masked_samples[:, 0] = torch.tensor(list(self.labels_map.values()))[rnd_labels]

        attention_mask = aux_data["attention_mask"]
        generation_data = {
            "attention_mask": attention_mask.repeat((self.batch_size, 1)).to(self.device)
        }

        for idx_step, idx_feature in enumerate(perturbable_inds.tolist()):
            num_batches = (num_samples + self.batch_size - 1) // self.batch_size

            for idx_batch in range(num_batches):
                s_batch, e_batch = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
                curr_input_ids = masked_samples[s_batch: e_batch]
                curr_batch_size = curr_input_ids.shape[0]

                res = self.generator(curr_input_ids.to(self.device),
                                     attention_mask=generation_data["attention_mask"][: curr_batch_size],
                                     return_dict=True)

                # ... - 1 due to LM: tokens < `i` generate `i`, so the 0th token is not getting generated
                curr_token_logits = res["logits"][:, idx_feature - 1]

                curr_preds = decoding_strategy(curr_token_logits,
                                               ensure_diff_from=None)

                masked_samples[torch.arange(s_batch, s_batch + curr_batch_size), idx_feature] = curr_preds[:, 0].cpu()

            # After selecting the first token, change label for controlled generation back to the predicted one
            if idx_step == 0:
                masked_samples[:, 0] = self.labels_map[self.possible_labels[label]]

        # TODO: once reworked, return weights corresponding to probabilities here
        sample_token_weights = torch.ones_like(masked_samples, dtype=torch.float32)

        return {
            "input_ids": masked_samples,
            "weights": sample_token_weights
        }


class GPTLMGenerator(SampleGenerator):
    def __init__(self, tokenizer_name, model_name, batch_size=2, max_seq_len=42, device="cuda",
                 top_p: Optional[float] = None):
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.top_p = top_p

        assert device in ["cpu", "cuda"]
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Device is set to 'cuda', but no CUDA device could be found. If you want to run the model "
                             "on CPU, set device to 'cpu'")
        self.device = torch.device(device)

        self.tokenizer = OpenAIGPTTokenizer.from_pretrained(self.tokenizer_name)
        self.generator = OpenAIGPTLMHeadModel.from_pretrained(self.model_name).to(self.device)
        self.generator.eval()

        # Required to build sequence: <BOS> <seq1> [<SEP> <seq2>] <EOS>
        assert self.tokenizer.bos_token_id is not None
        assert self.tokenizer.sep_token_id is not None
        assert self.tokenizer.eos_token_id is not None

    def from_internal(self, encoded_data: torch.Tensor) -> List[Union[str, Tuple[str, ...]]]:
        decoded_data = []
        for idx_example in range(encoded_data.shape[0]):
            sep_tokens = torch.nonzero(encoded_data[idx_example] == self.tokenizer.sep_token_id, as_tuple=False)

            # Multiple sequences present: <BOS> <seq1> <SEP> <seq2> <EOS> -> (<seq1>, <seq2>)
            if sep_tokens.shape[0] == 1:
                bnd = int(sep_tokens[0])
                seq1 = self.tokenizer.decode(encoded_data[idx_example, :bnd], skip_special_tokens=True)
                seq2 = self.tokenizer.decode(encoded_data[idx_example, bnd + 1:], skip_special_tokens=True)
                decoded_data.append((seq1, seq2))
            else:
                decoded_data.append(self.tokenizer.decode(encoded_data[idx_example], skip_special_tokens=True))

        return decoded_data

    def to_internal(self, text_data: List[Union[str, Tuple[str, ...]]]) -> Dict:
        res = self.tokenizer.batch_encode_plus(text_data, return_special_tokens_mask=True, return_tensors="pt",
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


class BertForMaskedLMGenerator(SampleGenerator):
    MLM_MAX_MASK_PROPORTION = 0.15

    def __init__(self, tokenizer_name, model_name, batch_size=8, max_seq_len=64, device="cuda",
                 strategy: Optional[str] = "top_k", top_p: Optional[float] = None, top_k: Optional[int] = 5,
                 threshold: Optional[float] = 0.1):
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.strategy = strategy
        assert self.strategy in ["top_k", "top_p", "threshold", "greedy"]
        self.top_p = top_p
        self.top_k = top_k
        self.threshold = threshold

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

    def from_internal(self, encoded_data):
        decoded_data = []
        for idx_example in range(encoded_data.shape[0]):
            sep_tokens = torch.nonzero(encoded_data[idx_example] == self.tokenizer.sep_token_id, as_tuple=False)

            # Multiple sequences present: [CLS] <seq1> [SEP] <seq2> [SEP] -> (<seq1>, <seq2>)
            if sep_tokens.shape[0] == 2:
                bnd = int(sep_tokens[0])
                seq1 = self.tokenizer.decode(encoded_data[idx_example, :bnd], skip_special_tokens=True)
                seq2 = self.tokenizer.decode(encoded_data[idx_example, bnd + 1:], skip_special_tokens=True)
                decoded_data.append((seq1, seq2))
            else:
                decoded_data.append(self.tokenizer.decode(encoded_data[idx_example], skip_special_tokens=True))

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
        num_features = int(input_ids.shape[1])

        perturbable_inds = torch.arange(num_features)[perturbable_mask[0]]
        num_perturbable = perturbable_inds.shape[0]

        _sequence_indexer = torch.arange(num_features).unsqueeze(0)  # used for selecting all tokens

        permutation_probas = []
        masked_samples = []

        token_type_ids = aux_data["token_type_ids"]
        attention_mask = aux_data["attention_mask"]

        generation_data = {
            "token_type_ids": token_type_ids.repeat((self.batch_size, 1)).to(self.device),
            "attention_mask": attention_mask.repeat((self.batch_size, 1)).to(self.device)
        }

        # Obtain guaranteed unique feature values by taking the topk predictions that are different from curr value
        for idx_feature in perturbable_inds:
            curr_masked = input_ids.clone()
            original_token = int(curr_masked[0, idx_feature])
            curr_masked[:, idx_feature] = self.tokenizer.mask_token_id

            curr_input_ids = curr_masked[:1]  # [B=1, num_features]
            res = self.generator(curr_input_ids.to(self.device),
                                 token_type_ids=generation_data["token_type_ids"][:1],
                                 attention_mask=generation_data["attention_mask"][:1],
                                 return_dict=True)
            logits = res["logits"][:, idx_feature]  # [1, |V|]
            logits[:, original_token] = -float("inf")

            if self.strategy == "threshold":
                probas = torch.softmax(logits, dim=-1)
                generated_tokens = torch.nonzero(probas > self.threshold, as_tuple=False)[:, 1]
            elif self.strategy == "top_k":
                _, indices = torch.topk(logits, k=self.top_k, sorted=False)
                generated_tokens = indices[0]
            elif self.strategy == "top_p":
                filtered_logits = top_p_filtering(logits, top_p=self.top_p)
                filtered_probas = torch.softmax(filtered_logits, dim=-1)
                generated_tokens = torch.nonzero(filtered_probas > 0, as_tuple=False)[:, 1]
            else:
                raise NotImplementedError(f"Unrecognized strategy: '{self.strategy}'")

            num_curr_generated = generated_tokens.shape[0]
            input_copy = input_ids.repeat((num_curr_generated, 1))
            input_copy[:, idx_feature] = generated_tokens
            masked_samples.append(input_copy)

            valid_inds_probas = torch.zeros((num_curr_generated, num_features), dtype=torch.float32)
            valid_inds_probas[:, perturbable_inds] = 1 / (num_perturbable - 1)
            valid_inds_probas[:, idx_feature] = 0.0
            permutation_probas.append(valid_inds_probas)

        masked_samples = torch.cat(masked_samples)  # [num_total_generated, max_seq_len]
        permutation_probas = torch.cat(permutation_probas, dim=0)
        permuted_indices = torch.multinomial(permutation_probas,
                                             num_samples=(num_perturbable - 1))  # [num_total_generated, num_perturbable]
        num_total_generated = masked_samples.shape[0]

        # Mask and predict all tokens, one token at a time, in different order - slightly diverse greedy decoding
        for idx_chunk in range(num_perturbable - 1):
            curr_masked = permuted_indices[:, idx_chunk: (idx_chunk + 1)]
            curr_mask_size = curr_masked.shape[1]
            masked_samples[torch.arange(num_total_generated).unsqueeze(1), curr_masked] = self.tokenizer.mask_token_id

            num_batches = (num_total_generated + self.batch_size - 1) // self.batch_size
            for idx_batch in range(num_batches):
                s_batch, e_batch = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
                curr_input_ids = masked_samples[s_batch: e_batch]
                curr_batch_size = curr_input_ids.shape[0]

                res = self.generator(curr_input_ids.to(self.device),
                                     token_type_ids=generation_data["token_type_ids"][: curr_batch_size],
                                     attention_mask=generation_data["attention_mask"][: curr_batch_size],
                                     return_dict=True)
                logits = res["logits"][torch.arange(curr_batch_size).unsqueeze(1), curr_masked[s_batch: e_batch]]
                for idx_token in range(curr_mask_size):
                    curr_token_logits = logits[:, idx_token]
                    curr_preds = greedy_decoding(curr_token_logits,
                                                 ensure_diff_from=None)

                    masked_samples[torch.arange(s_batch, s_batch + curr_batch_size),
                                   curr_masked[s_batch: e_batch, idx_token]] = curr_preds[:, 0].cpu()

        # Calculate the probabilities of tokens given their context (as given by BERT)
        num_batches = (num_total_generated + self.batch_size - 1) // self.batch_size
        mlm_token_probas = []
        for idx_batch in range(num_batches):
            s_batch, e_batch = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
            curr_input_ids = masked_samples[s_batch: e_batch]
            curr_batch_size = curr_input_ids.shape[0]
            _batch_indexer = torch.arange(curr_batch_size).reshape([-1, 1])

            res = self.generator(curr_input_ids.to(self.device),
                                 token_type_ids=generation_data["token_type_ids"][: curr_batch_size],
                                 attention_mask=generation_data["attention_mask"][: curr_batch_size],
                                 return_dict=True)
            probas = torch.softmax(res["logits"], dim=-1)
            mlm_token_probas.append(probas[_batch_indexer, _sequence_indexer, curr_input_ids])

        mlm_token_probas = torch.cat(mlm_token_probas)
        return {
            "input_ids": masked_samples,
            "weights": mlm_token_probas
        }


class BertForControlledMaskedLMGenerator(BertForMaskedLMGenerator):
    def __init__(self, tokenizer_name, model_name, control_labels: List[str], batch_size=8, max_seq_len=64, device="cuda",
                 strategy: Optional[str] = "greedy", top_p: Optional[float] = None, top_k: Optional[int] = 5,
                 threshold: Optional[float] = 0.1, label_weights: List = None, unique_dropout: Optional[float] = 0.0,
                 generate_expected_examples: Optional[bool] = False):
        super().__init__(tokenizer_name=tokenizer_name, model_name=model_name,
                         batch_size=batch_size, max_seq_len=max_seq_len, device=device,
                         strategy=strategy, top_p=top_p, top_k=top_k, threshold=threshold)
        self.unique_dropout = unique_dropout

        assert all(curr_control in self.tokenizer.all_special_tokens for curr_control in control_labels)
        self.control_labels = torch.tensor(self.tokenizer.encode(control_labels, add_special_tokens=False))

        self.label_weights = label_weights
        if self.label_weights is None:
            self.label_weights = [1.0] * len(self.control_labels)
        self.label_weights = torch.tensor(self.label_weights)

        # TODO: beam search = "global"/"sequence_level"?
        self.strategy_type = "token_level"
        if strategy == "greedy":
            self.decoding_strategy = lambda logits: greedy_decoding(logits)
        elif strategy == "top_p":
            assert self.top_p is not None
            self.decoding_strategy = lambda logits: top_p_decoding(logits, top_p=self.top_p)
        elif strategy == "top_k":
            assert self.top_k is not None
            self.decoding_strategy = lambda logits: top_k_decoding(logits, top_k=self.top_k)
        else:
            raise NotImplementedError(f"Unsupported decoding strategy: '{strategy}'")

        #  Denotes whether to change control label at every step in order to try generating "expected example"
        self.generate_expected = generate_expected_examples

    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor,
                 num_samples: Optional[int], **aux_data):
        # Make room for control label at start of sequence (at pos. 1)
        extended_input_ids = extend_tensor(input_ids).repeat((num_samples, 1))
        extended_pert_mask = extend_tensor(perturbable_mask)
        extended_token_type_ids = extend_tensor(aux_data["token_type_ids"]).repeat((self.batch_size, 1)).to(self.device)
        extended_attention_mask = extend_tensor(aux_data["attention_mask"]).repeat((self.batch_size, 1)).to(self.device)

        # Randomly select control labels for examples to be generated and set as attendable
        selected_control_labels = torch.multinomial(self.label_weights, num_samples=num_samples, replacement=True)
        selected_control_labels = self.control_labels[selected_control_labels]
        extended_input_ids[:, 1] = selected_control_labels
        extended_attention_mask[:, 1] = 1

        perturbable_inds = torch.arange(extended_input_ids.shape[1])[extended_pert_mask[0]]
        if self.strategy == "greedy":
            # If generation order was not shuffled, greedy decoding would likely produce identical samples
            weights = torch.zeros_like(extended_input_ids, dtype=torch.float32)
            weights[:, perturbable_inds] = 1
            generation_order = torch.multinomial(weights, num_samples=perturbable_inds.shape[0], replacement=False)
        else:
            generation_order = perturbable_inds.repeat((num_samples, 1))

        dropout_mask = torch.rand_like(generation_order, dtype=torch.float32) < self.unique_dropout
        if self.generate_expected:
            # Try approximating the expected example by switching control labels according to weights
            expanded_weights = self.label_weights.unsqueeze(0).repeat((num_samples, 1))
            control_labels = torch.multinomial(expanded_weights, num_samples=perturbable_inds.shape[0], replacement=True)
            control_labels = self.control_labels[control_labels]
        else:
            control_labels = selected_control_labels.unsqueeze(1).repeat((1, perturbable_inds.shape[0]))

        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for idx_batch in range(num_batches):
            s_b, e_b = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
            curr_input_ids = extended_input_ids[s_b: e_b]  # view
            curr_gen_order = generation_order[s_b: e_b]

            curr_batch_size = curr_input_ids.shape[0]
            batch_indexer = torch.arange(curr_batch_size)

            for i in range(curr_gen_order.shape[1]):
                curr_indices = curr_gen_order[:, i]
                orig_tokens = curr_input_ids[batch_indexer, curr_indices]
                curr_control_labels = control_labels[s_b: e_b, i]

                curr_input_ids[batch_indexer, curr_indices] = self.mask_token_id
                curr_input_ids[batch_indexer, 1] = curr_control_labels

                res = self.generator(input_ids=curr_input_ids.to(self.device),
                                     token_type_ids=extended_token_type_ids[:curr_batch_size],
                                     attention_mask=extended_attention_mask[:curr_batch_size])

                logits = res["logits"]  # [batch_size, max_seq_len, |V|]
                curr_masked_logits = logits[batch_indexer, curr_indices, :]

                curr_dropout_mask = dropout_mask[s_b: e_b, i]
                curr_masked_logits[batch_indexer[curr_dropout_mask], orig_tokens[curr_dropout_mask]] = - float("inf")

                preds = self.decoding_strategy(curr_masked_logits)
                curr_input_ids[batch_indexer, curr_indices] = preds[:, 0].cpu()

        valid_tokens = torch.ones_like(extended_pert_mask)
        valid_tokens[0, 1] = False

        return {
            "input_ids": extended_input_ids[:, valid_tokens[0]]
        }


if __name__ == "__main__":
    generator = BertForControlledMaskedLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/ime-lm/resources/weights/bert_snli_clm_best",
                                                   model_name="/home/matej/Documents/embeddia/interpretability/ime-lm/resources/weights/bert_snli_clm_best",
                                                   control_labels=["<ENTAILMENT>", "<NEUTRAL>", "<CONTRADICTION>"],
                                                   batch_size=2,
                                                   device="cpu")

    seq = ("A patient is being worked on by doctors and nurses", "A man is sleeping.")
    label = 0  # "entailment"
    encoded = generator.to_internal([seq])

    generated = generator.generate(encoded["input_ids"], label=label, perturbable_mask=encoded["perturbable_mask"],
                                   num_samples=5, **encoded["aux_data"])["input_ids"]

