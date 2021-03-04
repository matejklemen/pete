import json
import os

import torch


class UnigramModel:
    def __init__(self, logprobas: torch.Tensor,
                 max_length: int = None,
                 num_classes: int = None):
        self.logprobas = logprobas  # [[num_classes,] [max_length,] vocab_size]
        self.max_length = max_length
        self.num_classes = num_classes

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
                "max_length": self.max_length
            }, fp=f, indent=4)

        torch.save(self.logprobas, os.path.join(model_dir, "logprobas.th"))


if __name__ == "__main__":
    from explain_nlp.experimental.data import load_nli, TransformerSeqPairDataset, LABEL_TO_IDX
    from transformers import BertTokenizer
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

        model = UnigramModel(logprobas, num_classes=len(LABEL_TO_IDX["snli"]))
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

        model = UnigramModel(logprobas, max_length=args.max_length, num_classes=len(LABEL_TO_IDX["snli"]))
        model.save_pretrained(args.save_dir)
