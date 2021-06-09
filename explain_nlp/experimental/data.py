from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_TO_IDX = {
    "snli": {"entailment": 0, "neutral": 1, "contradiction": 2},
    "sentinews": {"neutral": 0, "negative": 1, "positive": 2},
    "imsypp": {0: 0, 1: 1},  # no-op, labels are pre-encoded
    "semeval5": {0: 0, 1: 1},  # no-op, labels are pre-encoded
    "24sata": {0: 0, 1: 1},  # no-op, labels are pre-encoded
    "imdb": {0: 0, 1: 1},  # no-op, labels are pre-encoded
    "qqp": {0: 0, 1: 1},  # no-op, labels are pre-encoded
    "sst-2": {0: 0, 1: 1},  # no-op, labels are pre-encoded
}
IDX_TO_LABEL = {dataset: {i: lbl for lbl, i in label_mapping.items()}
                for dataset, label_mapping in LABEL_TO_IDX.items()}
IDX_TO_LABEL["semeval5"] = {0: "clean", 1: "toxic"}
IDX_TO_LABEL["imsypp"] = {0: "clean", 1: "hateful"}
IDX_TO_LABEL["24sata"] = {0: "clean", 1: "hateful"}
IDX_TO_LABEL["imdb"] = {0: "negative", 1: "positive"}
IDX_TO_LABEL["qqp"] = {0: "different", 1: "duplicate"}
IDX_TO_LABEL["sst-2"] = {0: "negative", 1: "positive"}


class TransformerSeqPairDataset(Dataset):
    def __init__(self, input_ids, token_type_ids, attention_mask, special_tokens_mask, labels, max_seq_len: int = 41):
        self.input_ids = input_ids
        self.segments = token_type_ids
        self.attn_masks = attention_mask
        self.special_tokens_masks = special_tokens_mask
        self.labels = labels
        self.max_seq_len = max_seq_len

    @staticmethod
    def build(first: Iterable[str], second: Iterable[str], labels: Iterable[int],
              tokenizer, max_seq_len: int = 41):
        dataset_dict = tokenizer.batch_encode_plus(list(zip(first, second)), max_length=max_seq_len,
                                                   padding="max_length", truncation="longest_first",
                                                   return_special_tokens_mask=True, return_tensors="pt")
        dataset_dict["labels"] = torch.tensor(labels)
        return TransformerSeqPairDataset(**dataset_dict)

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
    def __init__(self, input_ids, token_type_ids, attention_mask, special_tokens_mask, labels, max_seq_len: int = 41):
        self.input_ids = input_ids
        self.segments = token_type_ids
        self.attn_masks = attention_mask
        self.special_tokens_masks = special_tokens_mask
        self.labels = labels
        self.max_seq_len = max_seq_len

    @staticmethod
    def build(sequences: Iterable[str], labels: Iterable[int],
              tokenizer, max_seq_len: int = 41):
        dataset_dict = tokenizer.batch_encode_plus(sequences, max_length=max_seq_len,
                                                   padding="max_length", truncation="longest_first",
                                                   return_special_tokens_mask=True, return_tensors="pt")
        dataset_dict["labels"] = torch.tensor(labels)
        return TransformerSeqDataset(**dataset_dict)

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


def load_imsypp(file_path, sample_size=None):
    return pd.read_csv(file_path, nrows=sample_size, quoting=1)


def load_imdb(file_path, sample_size=None):
    df = pd.read_csv(file_path, nrows=sample_size)
    df["review"] = df["review"].apply(lambda s: s.replace("<br />", ""))

    return df


def load_sst2(file_path, sample_size=None):
    return pd.read_csv(file_path, sep="\t", nrows=sample_size, header=0)


def load_qqp(file_path, sample_size=None):
    df = pd.read_csv(file_path, sep="\t", nrows=sample_size, encoding="utf-8", quoting=3)
    AVAILABLE_COLS = ["question1", "question2"]
    if "is_duplicate" in df.columns:
        AVAILABLE_COLS.append("is_duplicate")

    df = df.dropna(axis=0, how="any", subset=AVAILABLE_COLS)

    if "is_duplicate" in df.columns:
        df["is_duplicate"] = df["is_duplicate"].apply(lambda lbl: int(lbl))

    return df
