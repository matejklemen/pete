from typing import Tuple, Union, Dict, List, Optional
from warnings import warn

import torch
from transformers import BertTokenizer, BertForMaskedLM, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

from explain_nlp.methods.decoding import greedy_decoding, top_p_decoding, top_p_filtering


# TODO: mask, mask_token_id properties
# TODO: mask token in generator is not necessarily same as mask token in model (e.g. <MASK> vs [MASK])
class SampleGenerator:
    def from_internal(self, encoded_data: torch.Tensor) -> List[Union[str, Tuple[str, ...]]]:
        """ Convert from internal generator representation to text."""
        raise NotImplementedError

    def to_internal(self, text_data: List[Union[str, Tuple[str, ...]]],
                    labels: List[int]) -> Dict:
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

    def to_internal(self, text_data: List[Union[str, Tuple[str, ...]]], labels: List[int]) -> Dict:
        formatted_text = []
        for curr_text, curr_lbl in zip(text_data, labels):
            curr_lbl_str = self.possible_labels[curr_lbl]
            if isinstance(curr_text, str):
                formatted_text.append(f"<{curr_lbl_str.upper()}> {curr_text} {self.tokenizer.eos_token}")
            else:
                formatted_text.append(f"<{curr_lbl_str.upper()}> {curr_text[0]} {self.tokenizer.sep_token} "
                                      f"{curr_text[1]} {self.tokenizer.eos_token}")

        res = self.tokenizer.batch_encode_plus(formatted_text, return_special_tokens_mask=True, return_tensors="pt",
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

    def to_internal(self, text_data: List[Union[str, Tuple[str, ...]]],
                    labels: List[Union[int, str]] = None) -> Dict:
        formatted_text = []
        for curr_text in text_data:
            if isinstance(curr_text, str):
                formatted_text.append(f"{self.tokenizer.bos_token} {curr_text} {self.tokenizer.eos_token}")
            else:
                formatted_text.append(f"{self.tokenizer.bos_token} {curr_text[0]} {self.tokenizer.sep_token} "
                                      f"{curr_text[1]} {self.tokenizer.eos_token}")

        res = self.tokenizer.batch_encode_plus(formatted_text, return_special_tokens_mask=True, return_tensors="pt",
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
                 threshold: Optional[float] = 0.1, is_controlled_lm: Optional[bool] = False):
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.strategy = strategy
        assert self.strategy in ["top_k", "top_p", "threshold", "num_samples"]
        self.top_p = top_p
        self.top_k = top_k
        self.threshold = threshold

        self.is_controlled_lm = is_controlled_lm

        assert self.batch_size > 1 and self.batch_size % 2 == 0
        assert device in ["cpu", "cuda"]
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Device is set to 'cuda', but no CUDA device could be found. If you want to run the model "
                             "on CPU, set device to 'cpu'")
        self.device = torch.device(device)

        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        self.generator = BertForMaskedLM.from_pretrained(self.model_name).to(self.device)
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

    def to_internal(self, text_data, labels: List[Union[int, str]] = None):
        _text_data = text_data
        # Format examples in a controlled LM fashion (<LABEL> <seq1> [<seq2>])
        if self.is_controlled_lm and labels is not None:
            assert isinstance(labels[0], str)  # currently assuming label of example is passed in raw form
            assert len(text_data) == len(labels)

            control_formatted = []
            for curr_text, curr_lbl in zip(text_data, labels):
                if isinstance(curr_text, tuple):
                    control_formatted.append(f"{curr_lbl} {curr_text[0]} {curr_text[1]}")
                elif isinstance(curr_text, str):
                    control_formatted.append(f"{curr_lbl} {curr_text}")
            _text_data = control_formatted

        res = self.tokenizer.batch_encode_plus(_text_data, return_special_tokens_mask=True, return_tensors="pt",
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
                 num_samples: Optional[int] = -1, label=None, **aux_data) -> Dict:
        if num_samples > 0 and self.strategy != "num_samples":
            warn("Strategy is not set to 'num_samples', so the argument is being ignored")

        num_features = int(input_ids.shape[1])

        perturbable_inds = torch.arange(num_features)[perturbable_mask[0]]
        num_perturbable = perturbable_inds.shape[0]

        _sequence_indexer = torch.arange(num_features).unsqueeze(0)  # used for selecting all tokens

        # For each feature, reserve some amount of samples that are guaranteed to have that feature unique
        if self.strategy == "num_samples":
            # TODO: allocate samples accoring to estimated probabilities: store probas and take tokens that correspond
            #  to top `num_samples` probabilities
            uniq_samples_per_feature = torch.zeros(num_features, dtype=torch.long)
            uniq_samples_per_feature[perturbable_inds] = torch.floor_divide(num_samples, num_perturbable)
            rnd_selected = torch.randperm(num_perturbable)[:(num_samples % num_perturbable)]
            rnd_selected = perturbable_inds[rnd_selected]
            uniq_samples_per_feature[rnd_selected] += 1

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
            elif self.strategy == "num_samples":
                expected_num_gen = uniq_samples_per_feature[idx_feature]
                if expected_num_gen == 0:
                    continue

                _, indices = torch.topk(logits, k=expected_num_gen, sorted=False)
                generated_tokens = indices[0]
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


if __name__ == "__main__":
    generator = GPTControlledLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/ime-lm/examples/weights/gpt_snli_clm_maxseqlen42",
                                         model_name="/home/matej/Documents/embeddia/interpretability/ime-lm/examples/weights/gpt_snli_clm_maxseqlen42",
                                         possible_labels=["entailment", "neutral", "contradiction"],
                                         batch_size=2,
                                         max_seq_len=42,
                                         top_p=0.9,
                                         device="cpu")

    # generator = GPTLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/ime-lm/examples/weights/gpt_snli_lm_maxseqlen42",
    #                            model_name="/home/matej/Documents/embeddia/interpretability/ime-lm/examples/weights/gpt_snli_lm_maxseqlen42",
    #                            batch_size=2,
    #                            max_seq_len=42,
    #                            top_p=0.9,
    #                            device="cpu")
    # generator = BertForMaskedLMGenerator(tokenizer_name="bert-base-uncased",
    #                                      model_name="bert-base-uncased",
    #                                      batch_size=2,
    #                                      device="cpu")

    seq = ("A patient is being worked on by doctors and nurses", "A man is sleeping.")
    label = 0  # "entailment"
    encoded = generator.to_internal([seq], labels=[label])

    generator.generate(encoded["input_ids"], label=label, perturbable_mask=encoded["perturbable_mask"], num_samples=10,
                       **encoded["aux_data"])



