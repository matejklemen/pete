import argparse
import json
import logging
import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizerFast

from explain_nlp.experimental.data import load_imsypp, LABEL_TO_IDX, TransformerSeqDataset, IDX_TO_LABEL

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, default="/home/matej/Documents/data/imsypp/split/train.csv")
parser.add_argument("--dev_path", type=str, default="/home/matej/Documents/data/imsypp/split/dev.csv")
parser.add_argument("--test_path", type=str, default="/home/matej/Documents/data/imsypp/split/test.csv")

parser.add_argument("--model_save_dir", type=str, default="./imyspp_model",
                    help="Directory where a trained model is to be saved")
parser.add_argument("--pretrained_name_or_path", type=str, default="EMBEDDIA/crosloengual-bert",
                    help="Model to use as base for fine-tuning")
parser.add_argument("--learning_rate", type=float, default=2e-5)
# 95th percentile: 41, max: 81
parser.add_argument("--max_seq_len", type=int, default=41)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_epochs", type=int, default=2)
parser.add_argument("--validate_every_n_examples", type=int, default=5_000)
parser.add_argument("--early_stopping_rounds", type=int, default=5)


if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(args.model_save_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(os.path.join(args.model_save_dir, "train_log.txt")))

    logging.info(f"Using device: {DEVICE}")
    with open(os.path.join(args.model_save_dir, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), fp=f, indent=4)

    for k, v in vars(args).items():
        logging.info(f"|{k:30s}|{str(v):80s}|")

    train_df = load_imsypp(args.train_path)
    dev_df = load_imsypp(args.dev_path)
    test_df = load_imsypp(args.test_path)
    logging.info(f"Loaded {train_df.shape[0]} train, {dev_df.shape[0]} dev and {test_df.shape[0]} test examples")

    num_labels = len(LABEL_TO_IDX["imsypp"])
    logging.info(f"Using {num_labels} labels")

    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_name_or_path)
    model = BertForSequenceClassification.from_pretrained(args.pretrained_name_or_path,
                                                          num_labels=num_labels).to(DEVICE)

    logging.info("Constructing datasets")
    train_set = TransformerSeqDataset(sequences=train_df["preprocessed_tweet"].values.tolist(),
                                      labels=train_df["is_hate_speech"].values.tolist(),
                                      tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    dev_set = TransformerSeqDataset(sequences=dev_df["preprocessed_tweet"].values.tolist(),
                                    labels=dev_df["is_hate_speech"].values.tolist(),
                                    tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    test_set = TransformerSeqDataset(sequences=test_df["preprocessed_tweet"].values.tolist(),
                                     labels=test_df["is_hate_speech"].values.tolist(),
                                     tokenizer=tokenizer, max_seq_len=args.max_seq_len)

    # Copy over tokenizer so that model and tokenizer can later be loaded using same handle
    tokenizer.save_pretrained(args.model_save_dir)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    num_subsets = (len(train_set) + args.validate_every_n_examples - 1) // args.validate_every_n_examples
    best_dev_loss, rounds_no_increase = float("inf"), 0
    stop_training = False
    for idx_epoch in range(args.max_epochs):
        logging.info(f"Epoch #{idx_epoch}/{args.max_epochs - 1}")
        rand_indices = torch.randperm(len(train_set))
        tr_loss, num_processed = 0.0, 0

        # train & validate for every subset
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
            logging.info(f"[Subset#{idx_subset}] Training loss: {tr_loss / num_processed: .4f}")

            model.eval()
            dev_loss, dev_correct = 0.0, 0
            with torch.no_grad():
                for curr_batch in tqdm(DataLoader(dev_set, batch_size=args.batch_size)):
                    _curr_batch = {k: v.to(DEVICE) for k, v in curr_batch.items() if k != "special_tokens_mask"}
                    output = model(**_curr_batch, return_dict=True)
                    dev_loss += float(output["loss"])
                    dev_correct += int(torch.sum(torch.argmax(output["logits"], dim=1) == _curr_batch["labels"]))

            dev_loss /= (len(dev_set) / args.batch_size)
            dev_acc = dev_correct / len(dev_set)
            logging.info(f"[Subset#{idx_subset}] Dev loss: {dev_loss: .4f}, dev accuracy: {dev_acc: .4f}")

            if dev_loss < best_dev_loss:
                model.save_pretrained(args.model_save_dir)
                best_dev_loss = dev_loss
                rounds_no_increase = 0
            else:
                rounds_no_increase += 1

            if rounds_no_increase == args.early_stopping_rounds:
                logging.info(f"Stopping early because the dev accuracy did not increase for {args.early_stopping_rounds} rounds")
                logging.info(f"Best dev loss: {best_dev_loss: .4f}")
                stop_training = True
                break

        if stop_training:
            break

    logging.info("Evaluating model on test set")
    uniq, count = torch.unique(test_set.labels, return_counts=True)
    uniq = [IDX_TO_LABEL["sentinews"][int(_i)] for _i in uniq]
    count = count.float()
    count /= torch.sum(count)
    logging.info(f"[Diagnostics] Test set class distribution is: {dict(zip(uniq, count.tolist()))}")

    model.eval()
    test_correct = 0
    with torch.no_grad():
        for curr_batch in tqdm(DataLoader(test_set, batch_size=args.batch_size)):
            _curr_batch = {k: v.to(DEVICE) for k, v in curr_batch.items() if k != "special_tokens_mask"}
            output = model(**_curr_batch, return_dict=True)
            test_correct += int(torch.sum(torch.argmax(output["logits"], dim=1) == _curr_batch["labels"]))

    test_acc = test_correct / len(test_set)
    logging.info(f"Test accuracy: {test_acc: .4f}")
