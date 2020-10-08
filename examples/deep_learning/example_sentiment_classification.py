import argparse
import os
from time import time

import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

from explain_nlp.methods.ime import IMEExplainer, estimate_max_samples
from explain_nlp.methods.ime_mlm import IMEMaskedLMExplainer
from explain_nlp.visualizations.highlight import highlight_plot

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device {DEVICE}")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "interpret"], default="interpret")
parser.add_argument("--train_path", type=str, default="/home/matej/Documents/data/aclImdb/train/data.csv")
parser.add_argument("--test_path", type=str, default="/home/matej/Documents/data/aclImdb/test/data.csv")
parser.add_argument("--model_save_dir", type=str, default="./imdb_model",
                    help="Directory where a trained model is to be saved. Only used if --mode='train'")
parser.add_argument("--pretrained_name_or_path", type=str,
                    default="/home/matej/Documents/embeddia/interpretability/ime-lm/examples/weights/imdb_model",
                    help="Model to use as (a) base for fine-tuning (if --mode='train') or (b) model to interpret "
                         "(if --mode='interpret')")
parser.add_argument("--train_size", type=float, default=0.8)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--max_seq_len", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_epochs", type=int, default=5)
parser.add_argument("--validate_every_n_examples", type=int, default=5_000)
parser.add_argument("--early_stopping_rounds", type=int, default=5)


class ImdbDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_seq_len=128):
        self.input_ids = []
        self.segments = []
        self.attn_masks = []
        self.special_tokens_masks = []
        self.labels = []
        self.max_seq_len = max_seq_len

        for curr_review, curr_label in zip(reviews, labels):
            processed = tokenizer.encode_plus(curr_review, max_length=max_seq_len,
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


if __name__ == "__main__":
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_path, nrows=1_000)
    test_df = pd.read_csv(args.test_path, nrows=10)

    train_df["review"] = train_df["review"].apply(lambda s: s.replace("<br />", ""))
    test_df["review"] = test_df["review"].apply(lambda s: s.replace("<br />", ""))

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_name_or_path)
    model = BertForSequenceClassification.from_pretrained(args.pretrained_name_or_path).to(DEVICE)

    train_df, dev_df = train_test_split(train_df, train_size=args.train_size)
    train_set = ImdbDataset(reviews=train_df["review"].values,
                            labels=train_df["label"].values,
                            tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    dev_set = ImdbDataset(reviews=dev_df["review"],
                          labels=dev_df["label"],
                          tokenizer=tokenizer, max_seq_len=args.max_seq_len)

    # Train a model to be interpreted by re-running this script with --mode='interpret'
    if args.mode == "train":
        os.makedirs(args.model_save_dir, exist_ok=True)

        # Copy over tokenizer so that model and tokenizer can later be loaded using same handle
        tokenizer.save_pretrained(args.model_save_dir)

        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        num_subsets = (len(train_set) + args.validate_every_n_examples - 1) // args.validate_every_n_examples
        best_dev_acc, rounds_no_increase = 0.0, 0
        for idx_epoch in range(args.max_epochs):
            rand_indices = torch.randperm(len(train_set))
            tr_loss, num_processed = 0.0, 0

            # train - validate for every subset
            for idx_subset in range(num_subsets):
                curr_subset = Subset(train_set, rand_indices[idx_subset * args.validate_every_n_examples:
                                                             (idx_subset + 1) * args.validate_every_n_examples])
                model.train()
                for curr_batch in tqdm(DataLoader(curr_subset, batch_size=args.batch_size)):
                    _curr_batch = {k: v.to(DEVICE) for k, v in curr_batch.items() if k != "special_tokens_mask"}
                    output = model(**_curr_batch, return_dict=True)
                    curr_loss = output["loss"]
                    tr_loss += float(curr_loss)

                    curr_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                num_processed += len(curr_subset) / args.batch_size
                print(f"[Subset#{1 + idx_subset}] Training loss: {tr_loss / num_processed: .4f}")

                model.eval()
                dev_loss, dev_correct = 0.0, 0
                with torch.no_grad():
                    for curr_batch in DataLoader(dev_set, batch_size=args.batch_size):
                        _curr_batch = {k: v.to(DEVICE) for k, v in curr_batch.items() if k != "special_tokens_mask"}
                        output = model(**_curr_batch, return_dict=True)
                        dev_loss += float(output["loss"])
                        dev_correct += int(torch.sum(torch.argmax(output["logits"], dim=1) == _curr_batch["labels"]))

                dev_loss /= (len(dev_set) / args.batch_size)
                dev_acc = dev_correct / len(dev_set)
                print(f"[Subset#{1 + idx_subset}] Dev loss: {dev_loss: .4f}, dev accuracy: {dev_acc: .4f}")

                if dev_acc > best_dev_acc:
                    model.save_pretrained(args.model_save_dir)
                    best_dev_acc = dev_acc
                    rounds_no_increase = 0
                else:
                    rounds_no_increase += 1

                if rounds_no_increase == args.early_stopping_rounds:
                    print(f"Stopping early because the dev accuracy did not increase for {args.early_stopping_rounds} rounds")
                    exit(0)

    # Interpret an existing model
    else:
        test_set = ImdbDataset(reviews=test_df["review"].values,
                               labels=test_df["label"].values,
                               tokenizer=tokenizer, max_seq_len=args.max_seq_len)
        explainer = IMEExplainer(sample_data=train_set.input_ids)
        all_sequences = []
        all_labels = []
        all_importances = []

        i = 0

        for curr_example in DataLoader(test_set, batch_size=1, shuffle=False):
            @torch.no_grad()
            def _model_wrapper(data: torch.Tensor):
                num_examples = data.shape[0]
                batch_size = min(args.batch_size, num_examples)
                aux_inputs = {
                    "token_type_ids": curr_example["token_type_ids"].repeat_interleave(batch_size, dim=0).to(DEVICE),
                    "attention_mask": curr_example["attention_mask"].repeat_interleave(batch_size, dim=0).to(DEVICE)
                }

                num_batches = (num_examples + batch_size - 1) // batch_size
                all_scores = []
                for idx_batch in range(num_batches):
                    input_ids = data[idx_batch * batch_size: (idx_batch + 1) * batch_size]
                    curr_batch_size = input_ids.shape[0]
                    output = model(input_ids=input_ids.to(DEVICE),
                                   token_type_ids=aux_inputs["token_type_ids"][: curr_batch_size],
                                   attention_mask=aux_inputs["attention_mask"][: curr_batch_size],
                                   return_dict=True)
                    all_scores.append(torch.nn.functional.softmax(output["logits"], dim=-1))

                return torch.cat(all_scores, dim=0)


            _curr_example = {k: v.to(DEVICE) for k, v in curr_example.items() if k not in {"labels",
                                                                                           "special_tokens_mask"}}
            probas = _model_wrapper(_curr_example["input_ids"])
            predicted_label = int(torch.argmax(probas))
            t1 = time()
            explainer = IMEMaskedLMExplainer(model_func=_model_wrapper)
            imps, vars = explainer.explain(curr_example["input_ids"],
                                           label=predicted_label,
                                           min_samples_per_feature=10,
                                           perturbable_mask=torch.logical_not(curr_example["special_tokens_mask"]))
            t2 = time()
            all_sequences.append(tokenizer.convert_ids_to_tokens(curr_example["input_ids"][0]))
            all_importances.append(imps.tolist())
            all_labels.append(int(curr_example["labels"]))
            print(f"Time taken: {t2 - t1: .4f}s")

            required_samples = int(estimate_max_samples(vars * 10, alpha=0.95, max_abs_error=0.01))
            print(f"Required samples: {required_samples}")
            break

        highlight_plot(all_sequences, all_labels, all_importances, path="./visualize_mlm.html")
