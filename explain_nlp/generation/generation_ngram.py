import json
import os
import pickle
from collections import Counter
from typing import Optional, Mapping

import torch
from tqdm import tqdm
from transformers import BertTokenizer

from explain_nlp.generation.generation_base import SampleGenerator

# TODO: To be moved to some custom_modules package, once it exists
TrigramTokenizer = BertTokenizer


# TODO: To be moved to some custom_modules package, once it exists
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
    def __init__(self, tokenizer_name, model_name, max_seq_len, batch_size=8, device="cuda",
                 strategy="top_p", top_p=0.9, top_k=5, threshold=0.1,
                 generate_cover: Optional[bool] = False):
        super().__init__(max_seq_len=max_seq_len, batch_size=batch_size, device=device,
                         strategy=strategy, top_p=top_p, top_k=top_k, threshold=threshold)

        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.generate_cover = generate_cover

        assert self.batch_size > 1 and self.batch_size % 2 == 0
        assert device == "cpu", "Device 'cuda' is not implemented for trigram MLM, defaulting to 'cpu'"

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
