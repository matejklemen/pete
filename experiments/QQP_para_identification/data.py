import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_TO_IDX = {"clean": 0, "duplicate": 1}
IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}


class SequencePairDataset(Dataset):
    def __init__(self, q1, q2, labels, tokenizer, max_seq_len=55):
        self.input_ids = []
        self.segments = []
        self.attn_masks = []
        self.special_tokens_masks = []
        self.labels = []
        self.max_seq_len = max_seq_len

        for curr_q1, curr_q2, curr_label in zip(q1, q2, labels):
            processed = tokenizer.encode_plus(curr_q1, curr_q2, max_length=max_seq_len,
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


def load_qqp(file_path, sample_size=None):
    df = pd.read_csv(file_path, sep="\t", nrows=sample_size, encoding="utf-8", quoting=3)
    AVAILABLE_COLS = ["question1", "question2"]
    if "is_duplicate" in df.columns:
        AVAILABLE_COLS.append("is_duplicate")

    df = df.dropna(axis=0, how="any", subset=AVAILABLE_COLS)

    if "is_duplicate" in df.columns:
        df["is_duplicate"] = df["is_duplicate"].apply(lambda lbl: int(lbl))

    return df

