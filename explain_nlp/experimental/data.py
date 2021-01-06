from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_TO_IDX = {
    "snli": {"entailment": 0, "neutral": 1, "contradiction": 2},
    "sentinews": {"neutral": 0, "negative": 1, "positive": 2},
    "semeval5": {0: 0, 1: 1},  # no-op, labels are pre-encoded
    "24sata": {0: 0, 1: 1},  # no-op, labels are pre-encoded
    "imdb": {0: 0, 1: 1},  # no-op, labels are pre-encoded
    "qqp": {0: 0, 1: 1}  # no-op, labels are pre-encoded
}
IDX_TO_LABEL = {dataset: {i: lbl for lbl, i in label_mapping.items()}
                for dataset, label_mapping in LABEL_TO_IDX.items()}
IDX_TO_LABEL["semeval5"] = {0: "clean", 1: "toxic"}
IDX_TO_LABEL["24sata"] = {0: "clean", 1: "hateful"}
IDX_TO_LABEL["imdb"] = {0: "negative", 1: "positive"}
IDX_TO_LABEL["qqp"] = {0: "different", 1: "duplicate"}


class TransformerSeqPairDataset(Dataset):
    def __init__(self, first: Iterable[str], second: Iterable[str], labels: Iterable[int],
                 tokenizer, max_seq_len: int = 41):
        self.input_ids = []
        self.segments = []
        self.attn_masks = []
        self.special_tokens_masks = []
        self.labels = []
        self.max_seq_len = max_seq_len

        for seq1, seq2, curr_label in zip(first, second, labels):
            processed = tokenizer.encode_plus(seq1, seq2, max_length=max_seq_len,
                                              padding="max_length", truncation="longest_first",
                                              return_special_tokens_mask=True)
            self.input_ids.append(processed["input_ids"])
            self.segments.append(processed["token_type_ids"])
            self.attn_masks.append(processed["attention_mask"])
            self.special_tokens_masks.append(processed["special_tokens_mask"])
            self.labels.append(curr_label)

        self.input_ids = torch.tensor(self.input_ids)
        self.segments = torch.tensor(self.segments)
        self.attn_masks = torch.tensor(self.attn_masks)
        self.special_tokens_masks = torch.tensor(self.special_tokens_masks)
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "token_type_ids": self.segments[idx],
            "attention_mask": self.attn_masks[idx],
            "special_tokens_mask": self.special_tokens_masks[idx],
            "labels": self.labels[idx]
        }

    def __len__(self):
        return self.input_ids.shape[0]


class TransformerSeqDataset(Dataset):
    def __init__(self, sequences: Iterable[str], labels, tokenizer, max_seq_len=128):
        self.input_ids = []
        self.segments = []
        self.attn_masks = []
        self.special_tokens_masks = []
        self.labels = []
        self.max_seq_len = max_seq_len

        for curr_seq, curr_label in zip(sequences, labels):
            processed = tokenizer.encode_plus(curr_seq, max_length=max_seq_len,
                                              padding="max_length", truncation="longest_first",
                                              return_special_tokens_mask=True)
            self.input_ids.append(processed["input_ids"])
            self.segments.append(processed["token_type_ids"])
            self.attn_masks.append(processed["attention_mask"])
            self.special_tokens_masks.append(processed["special_tokens_mask"])
            self.labels.append(curr_label)

        self.input_ids = torch.tensor(self.input_ids)
        self.segments = torch.tensor(self.segments)
        self.attn_masks = torch.tensor(self.attn_masks)
        self.special_tokens_masks = torch.tensor(self.special_tokens_masks)
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "token_type_ids": self.segments[idx],
            "attention_mask": self.attn_masks[idx],
            "special_tokens_mask": self.special_tokens_masks[idx],
            "labels": self.labels[idx]
        }

    def __len__(self):
        return self.input_ids.shape[0]


def load_nli(file_path, sample_size=None):
    """ Common loader for SNLI/MultiNLI """
    df = pd.read_csv(file_path, sep="\t", na_values=[""], nrows=sample_size, encoding="utf-8", quoting=3)
    # Drop examples where one of the sentences is "n/a"
    df = df.dropna(axis=0, how="any", subset=["sentence1", "sentence2"])
    mask = df["gold_label"] != "-"
    df = df.loc[mask].reset_index(drop=True)

    return df


def load_sentinews(file_path, sample_size=None):
    return pd.read_csv(file_path, sep="\t", nrows=sample_size)


def load_semeval5(file_path, sample_size=None):
    return pd.read_csv(file_path, nrows=sample_size, encoding="utf-8")


def load_24sata(file_path, sample_size=None):
    return pd.read_csv(file_path, nrows=sample_size)


def load_imdb(file_path, sample_size=None):
    df = pd.read_csv(file_path, nrows=sample_size)
    df["review"] = df["review"].apply(lambda s: s.replace("<br />", ""))

    return df


def load_qqp(file_path, sample_size=None):
    df = pd.read_csv(file_path, sep="\t", nrows=sample_size, encoding="utf-8", quoting=3)
    AVAILABLE_COLS = ["question1", "question2"]
    if "is_duplicate" in df.columns:
        AVAILABLE_COLS.append("is_duplicate")

    df = df.dropna(axis=0, how="any", subset=AVAILABLE_COLS)

    if "is_duplicate" in df.columns:
        df["is_duplicate"] = df["is_duplicate"].apply(lambda lbl: int(lbl))

    return df
