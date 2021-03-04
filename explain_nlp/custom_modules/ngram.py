import json
import os
import pickle
from collections import Counter
from typing import Optional, Mapping, List, Dict

import torch
from tqdm import tqdm
from transformers import BertTokenizer

UnigramTokenizer = BertTokenizer
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


class UnigramModel:
    def __init__(self, logprobas: torch.Tensor,
                 max_length: int = None,
                 classes: Dict[str, int] = None):
        self.logprobas = logprobas  # [[num_classes,] [max_length,] vocab_size]
        self.max_length = max_length
        self.classes = classes

    @staticmethod
    def from_pretrained(model_dir):
        with open(os.path.join(model_dir, "unigram_config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        loaded_logprobas = torch.load(os.path.join(model_dir, "logprobas.th"))
        instance = UnigramModel(logprobas=loaded_logprobas, **config)

        return instance

    def save_pretrained(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        with open(os.path.join(model_dir, "unigram_config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "max_length": self.max_length,
                "classes": self.classes
            }, fp=f, indent=4)

        torch.save(self.logprobas, os.path.join(model_dir, "logprobas.th"))


if __name__ == "__main__":
    from explain_nlp.experimental.data import load_nli, TransformerSeqPairDataset, LABEL_TO_IDX
    from time import time
    import logging
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--used_model", type=str, choices=["unigram", "controlled_unigram",
                                                           "positional_unigram", "controlled_positional_unigram"],
                        default="controlled_positional_unigram")
    parser.add_argument("--training_path", type=str, default="/home/matej/Documents/data/snli/snli_1.0_train.txt")
    parser.add_argument("--max_length", type=int, default=41)

    parser.add_argument("--bert_tokenizer_handle", type=str, default="bert-base-uncased",
                        help="Handle of BERT tokenizer whose vocabulary is used in the modeling process")
    parser.add_argument("--save_dir", type=str, default="controlled_positional_unigram_lm_snli")

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_handle)
    tokenizer.save_pretrained(args.save_dir)
    with open(os.path.join(args.save_dir, "training_settings.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), fp=f, indent=4)

    df_train = load_nli(args.training_path)
    t_s = time()
    train_set = TransformerSeqPairDataset.build(first=df_train["sentence1"].values,
                                                second=df_train["sentence2"].values,
                                                labels=df_train["gold_label"].apply(
                                                    lambda label_str: LABEL_TO_IDX["snli"][label_str]
                                                ).values,
                                                tokenizer=tokenizer, max_seq_len=args.max_length)
    t_e = time()
    logging.info(f"Building dataset ({len(train_set)} examples) took {t_e - t_s: .4f}s")

    control_tokens = [f"<{lbl.upper()}>" for lbl in LABEL_TO_IDX["snli"]]
    control_to_idx = {lbl: i for i, lbl in enumerate(control_tokens)}

    if args.used_model == "unigram":
        count_tensor = torch.zeros(len(tokenizer))
        t_s = time()
        uniq_val, uniq_counts = torch.unique(train_set.input_ids, return_counts=True)
        t_e = time()
        logging.info(f"Frequency calculation took {t_e - t_s: .4f}s")

        count_tensor[uniq_val] = uniq_counts.float()
        count_tensor[tokenizer.pad_token_id] = 0
        count_tensor[tokenizer.cls_token_id] = 0
        count_tensor[tokenizer.sep_token_id] = 0

        sum_counts = torch.sum(count_tensor)
        logprobas = torch.log(count_tensor) - torch.log(sum_counts)

        model = UnigramModel(logprobas)
        model.save_pretrained(args.save_dir)
    elif args.used_model == "controlled_unigram":
        tokenizer.add_special_tokens({
            "additional_special_tokens": control_tokens
        })
        tokenizer.save_pretrained(args.save_dir)

        count_tensor = torch.zeros((len(LABEL_TO_IDX["snli"]), len(tokenizer)))
        t_s = time()
        for label, encoded_label in LABEL_TO_IDX["snli"].items():
            mask = train_set.labels == encoded_label
            curr_inputs = train_set.input_ids[mask]

            uniq_val, uniq_counts = torch.unique(curr_inputs, return_counts=True)
            count_tensor[encoded_label, uniq_val] = uniq_counts.float()
            count_tensor[encoded_label, tokenizer.pad_token_id] = 0
            count_tensor[encoded_label, tokenizer.cls_token_id] = 0
            count_tensor[encoded_label, tokenizer.sep_token_id] = 0

            if torch.sum(count_tensor[encoded_label, :]) == 0:
                print("Sum is 0")

        t_e = time()
        logging.info(f"Frequency calculation took {t_e - t_s: .4f}s")
        positional_sum = torch.sum(count_tensor, dim=1)
        logprobas = torch.log(count_tensor) - torch.log(positional_sum.unsqueeze(1))

        model = UnigramModel(logprobas, classes=control_to_idx)
        model.save_pretrained(args.save_dir)
    elif args.used_model == "positional_unigram":
        count_tensor = torch.zeros((args.max_length, len(tokenizer)))

        t_s = time()
        for idx_position in range(args.max_length):
            uniq_val, uniq_counts = torch.unique(train_set.input_ids[:, idx_position], return_counts=True)
            count_tensor[idx_position, uniq_val] = uniq_counts.float()
            count_tensor[idx_position, tokenizer.pad_token_id] = 0
            count_tensor[idx_position, tokenizer.cls_token_id] = 0
            count_tensor[idx_position, tokenizer.sep_token_id] = 0

        t_e = time()
        logging.info(f"Frequency calculation took {t_e - t_s: .4f}s")

        positional_sum = torch.sum(count_tensor, dim=1)
        logprobas = torch.log(count_tensor) - torch.log(positional_sum.unsqueeze(1))

        model = UnigramModel(logprobas, max_length=args.max_length)
        model.save_pretrained(args.save_dir)
    elif args.used_model == "controlled_positional_unigram":
        tokenizer.add_special_tokens({
            "additional_special_tokens": control_tokens
        })
        tokenizer.save_pretrained(args.save_dir)
        count_tensor = torch.zeros((len(LABEL_TO_IDX["snli"]), args.max_length, len(tokenizer)))

        t_s = time()
        for label, encoded_label in LABEL_TO_IDX["snli"].items():
            mask = train_set.labels == encoded_label
            curr_inputs = train_set.input_ids[mask]

            for idx_position in range(args.max_length):
                uniq_val, uniq_counts = torch.unique(curr_inputs[:, idx_position], return_counts=True)
                count_tensor[encoded_label, idx_position, uniq_val] = uniq_counts.float()
                count_tensor[encoded_label, idx_position, tokenizer.pad_token_id] = 0
                count_tensor[encoded_label, idx_position, tokenizer.cls_token_id] = 0
                count_tensor[encoded_label, idx_position, tokenizer.sep_token_id] = 0

                if torch.sum(count_tensor[encoded_label, idx_position, :]) == 0:
                    print("Sum is 0")

        t_e = time()
        logging.info(f"Frequency calculation took {t_e - t_s: .4f}s")

        positional_sum = torch.sum(count_tensor, dim=2)
        logprobas = torch.log(count_tensor) - torch.log(positional_sum.unsqueeze(2))

        model = UnigramModel(logprobas, max_length=args.max_length, classes=control_to_idx)
        model.save_pretrained(args.save_dir)
