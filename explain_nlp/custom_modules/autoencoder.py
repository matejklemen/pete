import json
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from transformers import BertTokenizerFast

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Use BERT's vocabulary as a slight simplification
LSTMAutoencoderSubwordTokenizer = BertTokenizerFast


class LSTMAutoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, max_length, dropout=0.0, bidirectional=False,
                 padding_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx
        self.max_length = max_length

        self.num_directions = 2 if bidirectional else 1

        self.embedder = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim,
                                     padding_idx=self.padding_idx)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                            bidirectional=self.bidirectional, batch_first=True)
        self.decoders = nn.ModuleList([nn.Linear(in_features=self.num_directions*self.hidden_size,
                                                 out_features=self.vocab_size) for _ in range(self.max_length)])

    @staticmethod
    def from_pretrained(model_dir):
        with open(os.path.join(model_dir, "lstmae_config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

        instance = LSTMAutoencoder(**config)
        instance.load_state_dict(torch.load(os.path.join(model_dir, "lstmae.th"), map_location=DEVICE))
        return instance

    def save_pretrained(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        with open(os.path.join(model_dir, "lstmae_config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "embedding_dim": self.embedding_dim,
                "hidden_size": self.hidden_size,
                "max_length": self.max_length,
                "dropout": self.dropout,
                "bidirectional": self.bidirectional,
                "padding_idx": self.padding_idx
            }, fp=f, indent=4)

        torch.save(self.state_dict(), os.path.join(model_dir, "lstmae.th"))

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        batch_size, max_len = input_ids.shape
        _batch_indexer = torch.arange(batch_size)

        eff_attention_mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask
        input_len = torch.sum(eff_attention_mask, dim=1)

        embedded_input = torch.dropout(self.embedder(input_ids),
                                       p=self.dropout,
                                       train=self.training)  # [batch_size, max_length, embedding_size]
        features, _ = self.lstm(embedded_input)  # [batch_size, max_length, 2 * hidden_size]
        decomposed_features = features.view(batch_size, max_len, self.num_directions, self.hidden_size)

        latent_repr = decomposed_features[_batch_indexer, input_len - 1, 0]  # [batch_size, hidden_size]
        if self.num_directions == 2:
            latent_repr = torch.cat((latent_repr, decomposed_features[:, 0, 1]), dim=1)  # [batch_size, 2 * hidden_size]

        latent_repr = torch.dropout(latent_repr,
                                    p=self.dropout,
                                    train=self.training)

        logits = []
        for idx_feature in range(self.max_length):
            curr_logits = self.decoders[idx_feature](latent_repr)
            logits.append(curr_logits)

        logits = torch.stack(logits, dim=1)  # [batch_size, max_length, vocab_size]

        ret = {"logits": logits}
        if labels is not None:
            criterion = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
            loss = criterion(logits.view(-1, self.vocab_size), labels.view(-1))
            ret["loss"] = loss

        return ret


if __name__ == "__main__":
    from explain_nlp.experimental.data import load_nli, TransformerSeqPairDataset, LABEL_TO_IDX
    import torch.optim as optim
    import logging
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--training_path", type=str, required=True)
    parser.add_argument("--validation_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="lstm_ae_lm")

    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--bert_tokenizer_handle", type=str, default="bert-base-uncased",
                        help="Handle of BERT tokenizer whose vocabulary is used in the modeling process")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=41)

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--validate_every_n_steps", type=int, default=5000)
    parser.add_argument("--early_stopping_rounds", type=int, default=5)

    args = parser.parse_args()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    tokenizer = LSTMAutoencoderSubwordTokenizer.from_pretrained(args.bert_tokenizer_handle)
    tokenizer.save_pretrained(args.save_dir)

    df_train = load_nli(args.training_path)
    df_dev = load_nli(args.validation_path)

    train_dataset = TransformerSeqPairDataset.build(first=df_train["sentence1"].values,
                                                    second=df_train["sentence2"].values,
                                                    labels=df_train["gold_label"].apply(
                                                        lambda label_str: LABEL_TO_IDX["snli"][label_str]
                                                    ).values,
                                                    tokenizer=tokenizer, max_seq_len=args.max_length)

    dev_dataset = TransformerSeqPairDataset.build(first=df_dev["sentence1"].values,
                                                  second=df_dev["sentence2"].values,
                                                  labels=df_dev["gold_label"].apply(
                                                      lambda label_str: LABEL_TO_IDX["snli"][label_str]
                                                  ).values,
                                                  tokenizer=tokenizer, max_seq_len=args.max_length)

    model = LSTMAutoencoder(vocab_size=len(tokenizer),
                            embedding_dim=args.embedding_size,
                            hidden_size=args.hidden_size,
                            max_length=args.max_length,
                            dropout=args.dropout,
                            bidirectional=args.bidirectional,
                            padding_idx=tokenizer.pad_token_id).to(DEVICE)
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)

    with open(os.path.join(args.save_dir, "training_settings.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), fp=f, indent=4)

    num_train_subsets = (len(train_dataset) + args.validate_every_n_steps - 1) // args.validate_every_n_steps
    best_dev_loss, no_increase = float("inf"), 0
    for idx_epoch in range(args.num_epochs):
        logging.info(f"*Epoch #{idx_epoch}*")
        training_loss, train_denom = 0.0, 0

        rand_indices = torch.randperm(len(train_dataset))
        for idx_subset in range(num_train_subsets):
            logging.info(f"Running subset #{idx_subset}")
            s_sub, e_sub = idx_subset * args.validate_every_n_steps, (idx_subset + 1) * args.validate_every_n_steps

            model.train()
            for curr_batch in DataLoader(Subset(train_dataset, rand_indices[s_sub: e_sub]),
                                         batch_size=args.batch_size):
                res = model(input_ids=curr_batch["input_ids"].to(DEVICE),
                            labels=curr_batch["input_ids"].to(DEVICE))

                curr_loss = res["loss"]
                training_loss += float(curr_loss)
                train_denom += 1

                curr_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            logging.info(f"Training loss: {training_loss / train_denom: .4f}")

            dev_loss, dev_denom = 0.0, 0
            model.eval()
            with torch.no_grad():
                for curr_dev_batch in DataLoader(dev_dataset, batch_size=2*args.batch_size):
                    res = model(input_ids=curr_dev_batch["input_ids"].to(DEVICE),
                                labels=curr_dev_batch["input_ids"].to(DEVICE))
                    dev_loss += float(res["loss"])
                    dev_denom += 1

            dev_loss /= dev_denom
            logging.info(f"Dev loss: {dev_loss: .4f}")
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                no_increase = 0

                logging.info("Saving new best model!")
                model.save_pretrained(args.save_dir)
            else:
                no_increase += 1

            if no_increase == args.early_stopping_rounds:
                logging.info(f"Stopping early... Best dev loss: {best_dev_loss: .4f}")
                exit(0)

