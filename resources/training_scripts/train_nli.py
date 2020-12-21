import json

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

import argparse
import os
import logging

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, default="/home/matej/Documents/data/snli/snli_1.0_train.txt")
parser.add_argument("--dev_path", type=str, default="/home/matej/Documents/data/snli/snli_1.0_dev.txt")
parser.add_argument("--test_path", type=str, default="/home/matej/Documents/data/snli/snli_1.0_test.txt")
parser.add_argument("--model_save_dir", type=str, default="./nli_snli_model",
                    help="Directory where a trained model is to be saved. Only used if --mode='train'")
parser.add_argument("--pretrained_name_or_path", type=str,
                    default="bert-base-uncased",
                    help="Model to use as (a) base for fine-tuning (if --mode='train') or (b) model to interpret "
                         "(if --mode='interpret')")
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--max_seq_len", type=int, default=41)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--max_epochs", type=int, default=5)
parser.add_argument("--validate_every_n_examples", type=int, default=5_000)
parser.add_argument("--early_stopping_rounds", type=int, default=5)


class NLIDataset(Dataset):
    def __init__(self, premises, hypotheses, labels, tokenizer, max_seq_len=41):
        self.input_ids = []
        self.segments = []
        self.attn_masks = []
        self.special_tokens_masks = []
        self.labels = []
        self.max_seq_len = max_seq_len

        for curr_premise, curr_hypothesis, curr_label in zip(premises, hypotheses, labels):
            processed = tokenizer.encode_plus(curr_premise, curr_hypothesis, max_length=max_seq_len,
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


if __name__ == "__main__":
    args = parser.parse_args()
    LABEL_TO_IDX = {"entailment": 0, "neutral": 1, "contradiction": 2}

    os.makedirs(args.model_save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.model_save_dir, f"train.log")),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Using device: {DEVICE}")
    with open(os.path.join(args.model_save_dir, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), fp=f, indent=4)

    num_labels = len(LABEL_TO_IDX)
    logging.info(f"Using {num_labels} labels")
    model = BertForSequenceClassification.from_pretrained(args.pretrained_name_or_path,
                                                          num_labels=num_labels).to(DEVICE)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_name_or_path)

    df_train = load_nli(args.train_path)
    df_dev = load_nli(args.dev_path)
    df_test = load_nli(args.test_path)
    logging.info(f"Loaded {df_train.shape[0]} train, {df_dev.shape[0]} dev and {df_test.shape[0]} test examples")

    logging.info("Constructing datasets")
    train_set = NLIDataset(premises=df_train["sentence1"].values,
                           hypotheses=df_train["sentence2"].values,
                           labels=df_train["gold_label"].apply(lambda label_str: LABEL_TO_IDX[label_str]).values,
                           tokenizer=tokenizer,
                           max_seq_len=args.max_seq_len)
    dev_set = NLIDataset(premises=df_dev["sentence1"].values,
                         hypotheses=df_dev["sentence2"].values,
                         labels=df_dev["gold_label"].apply(lambda label_str: LABEL_TO_IDX[label_str]).values,
                         tokenizer=tokenizer,
                         max_seq_len=args.max_seq_len)
    test_set = NLIDataset(premises=df_test["sentence1"].values,
                          hypotheses=df_test["sentence2"].values,
                          labels=df_test["gold_label"].apply(lambda label_str: LABEL_TO_IDX[label_str]).values,
                          tokenizer=tokenizer,
                          max_seq_len=args.max_seq_len)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    tokenizer.save_pretrained(args.model_save_dir)
    best_dev_acc, no_increase = 0.0, 0
    stop_training = False

    for idx_epoch in range(args.max_epochs):
        shuffled_indices = torch.randperm(len(train_set))

        num_minisets = (len(train_set) + args.validate_every_n_examples - 1) // args.validate_every_n_examples
        for idx_miniset in range(num_minisets):
            logging.info(f"Miniset {idx_miniset}/{num_minisets - 1}")
            curr_subset = Subset(train_set, shuffled_indices[idx_miniset * args.validate_every_n_examples:
                                                             (idx_miniset + 1) * args.validate_every_n_examples])
            num_sub_batches = (len(curr_subset) + args.batch_size - 1) // args.batch_size
            model.train()

            train_loss = 0.0
            for curr_batch in DataLoader(curr_subset, shuffle=False, batch_size=args.batch_size):
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
            dev_loss = 0.0
            num_correct = 0
            with torch.no_grad():
                model.eval()
                for curr_batch in DataLoader(dev_set, shuffle=False, batch_size=args.batch_size):
                    _curr_batch = {k: v.to(DEVICE) for k, v in curr_batch.items() if k != "special_tokens_mask"}
                    res = model(**_curr_batch, return_dict=True)
                    curr_loss = res["loss"]
                    num_correct += int(torch.sum(torch.argmax(res["logits"], dim=1) == _curr_batch["labels"]))

                    dev_loss += float(curr_loss)

            dev_acc = num_correct / len(dev_set)
            dev_loss = dev_loss / num_dev_batches
            logging.info(f"Validation accuracy = {dev_acc: .4f}")
            if dev_acc > best_dev_acc:
                logging.info("New best! Saving checkpoint")
                best_dev_acc = dev_acc
                no_increase = 0
                model.save_pretrained(args.model_save_dir)
            else:
                no_increase += 1

            if no_increase == args.early_stopping_rounds:
                logging.info(f"Stopping early after validation loss did not improve for {args.early_stopping_rounds} rounds")
                stop_training = True
                break

        if stop_training:
            break

    logging.info("Evaluating model on test set")
    num_correct_test = 0
    with torch.no_grad():
        model.eval()
        for curr_batch in DataLoader(test_set, shuffle=False, batch_size=args.batch_size):
            _curr_batch = {k: v.to(DEVICE) for k, v in curr_batch.items() if k != "special_tokens_mask"}
            res = model(**_curr_batch, return_dict=True)
            curr_loss = res["loss"]
            num_correct_test += int(torch.sum(torch.argmax(res["logits"], dim=1) == _curr_batch["labels"]))

    test_acc = num_correct_test / len(test_set)
    logging.info(f"Test accuracy: {test_acc: .4f}")
