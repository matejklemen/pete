import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

import argparse
import os
import logging
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device {DEVICE}")

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, default="/home/matej/Documents/data/QQP/train.tsv")
parser.add_argument("--dev_path", type=str, default="/home/matej/Documents/data/QQP/dev.tsv")
parser.add_argument("--test_path", type=str, default="/home/matej/Documents/data/QQP/test.tsv")
parser.add_argument("--model_save_dir", type=str, default="./qqp_model",
                    help="Directory where a trained model is to be saved")
parser.add_argument("--pretrained_name_or_path", type=str, default="bert-base-uncased",
                    help="Model to use as base for fine-tuning")
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--max_seq_len", type=int, default=55)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--max_epochs", type=int, default=5)
parser.add_argument("--validate_every_n_examples", type=int, default=20_000)
parser.add_argument("--early_stopping_rounds", type=int, default=5)


class SequencePairDataset(Dataset):
    def __init__(self, q1, q2, labels, tokenizer, max_seq_len=41):
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


if __name__ == "__main__":
    args = parser.parse_args()
    LABEL_TO_IDX = {"clean": 0, "duplicate": 1}
    IDX_TO_LABEL = {i: lbl for lbl, i in LABEL_TO_IDX.items()}

    os.makedirs(args.model_save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.model_save_dir, f"train.log")),
            logging.StreamHandler()
        ]
    )

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_name_or_path)

    df_train = load_qqp(args.train_path)
    df_dev = load_qqp(args.dev_path)

    majority_preds = np.zeros(df_dev.shape[0])
    majority_preds[:] = df_dev["is_duplicate"].value_counts().argmax()
    majority_acc = accuracy_score(df_dev["is_duplicate"].values, majority_preds)
    majority_f1 = f1_score(df_dev["is_duplicate"].values, majority_preds, average="macro")
    logging.info(f"Majority dev accuracy: {majority_acc: .4f}, majority F1: {majority_f1: .4f}")

    train_set = SequencePairDataset(q1=df_train["question1"].values,
                                    q2=df_train["question2"].values,
                                    labels=df_train["is_duplicate"].values,
                                    tokenizer=tokenizer,
                                    max_seq_len=args.max_seq_len)
    dev_set = SequencePairDataset(q1=df_dev["question1"].values,
                                  q2=df_dev["question2"].values,
                                  labels=df_dev["is_duplicate"].values,
                                  tokenizer=tokenizer,
                                  max_seq_len=args.max_seq_len)

    model = BertForSequenceClassification.from_pretrained(args.pretrained_name_or_path).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    tokenizer.save_pretrained(args.model_save_dir)
    best_dev_f1, dev_acc, no_increase = 0.0, 0.0, 0

    for idx_epoch in range(args.max_epochs):
        logging.info(f"Epoch #{idx_epoch}/{args.max_epochs - 1}")
        shuffled_indices = torch.randperm(len(train_set))

        num_minisets = (len(train_set) + args.validate_every_n_examples - 1) // args.validate_every_n_examples
        for idx_miniset in range(num_minisets):
            logging.info(f"Miniset {idx_miniset}/{num_minisets - 1}")
            curr_subset = Subset(train_set, shuffled_indices[idx_miniset * args.validate_every_n_examples:
                                                             (idx_miniset + 1) * args.validate_every_n_examples])
            num_sub_batches = (len(curr_subset) + args.batch_size - 1) // args.batch_size
            model.train()

            train_loss = 0.0
            for curr_batch in tqdm(DataLoader(curr_subset, shuffle=False, batch_size=args.batch_size)):
                _curr_batch = {k: v.to(DEVICE) for k, v in curr_batch.items() if k != "special_tokens_mask"}
                res = model(**_curr_batch, return_dict=True)
                curr_loss = res["loss"]

                curr_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += float(curr_loss)

            train_loss = train_loss / num_sub_batches
            logging.info(f"Training loss = {train_loss: .4f}")

            num_dev_batches = (len(dev_set) + args.batch_size - 1) // args.batch_size
            dev_preds = []
            with torch.no_grad():
                model.eval()
                for curr_batch in tqdm(DataLoader(dev_set, shuffle=False, batch_size=args.batch_size)):
                    _curr_batch = {k: v.to(DEVICE) for k, v in curr_batch.items() if k != "special_tokens_mask"}
                    res = model(**_curr_batch, return_dict=True)
                    curr_loss = res["loss"]
                    curr_preds = torch.argmax(res["logits"], dim=1)
                    dev_preds.append(curr_preds.cpu().numpy())

            dev_preds = np.concatenate(dev_preds)
            dev_acc = accuracy_score(dev_set.labels.cpu().numpy(), dev_preds)
            dev_f1 = f1_score(dev_set.labels.cpu().numpy(), dev_preds, average="macro")
            logging.info(f"Dev F1: {dev_f1: .4f}, accuracy = {dev_acc: .4f}")
            if dev_f1 > best_dev_f1:
                logging.info("New best! Saving checkpoint")
                best_dev_f1 = dev_f1
                no_increase = 0
                model.save_pretrained(args.model_save_dir)
            else:
                no_increase += 1

            if no_increase == args.early_stopping_rounds:
                logging.info(f"Stopping early after validation metric did not improve for {args.early_stopping_rounds} rounds")
                logging.info(f"Best dev F1: {best_dev_f1: .4f}")
                exit(0)  # TODO: replace this with breaks (so that we can eval on test set)
