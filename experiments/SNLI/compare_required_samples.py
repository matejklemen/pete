import argparse
import json
from time import time
import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.methods.ime_mlm import IMEMaskedLMExplainer
from explain_nlp.methods.utils import estimate_max_samples
from explain_nlp.visualizations.highlight import highlight_plot

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
EXPERIMENT_DESCRIPTION = \
"""Compare estimated number of required samples (IME vs IME+MLM) to satisfy theoretical 'guarantees' on max AE 
specified by `confidence_interval` and `max_abs_error`. Ran on SNLI. See config.json for specific options."""

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, default="/home/matej/Documents/data/snli/snli_1.0_train.txt")
parser.add_argument("--test_path", type=str, default="/home/matej/Documents/data/snli/snli_1.0_test.txt")
parser.add_argument("--model_dir", type=str, default="/home/matej/Documents/embeddia/interpretability/ime-lm/examples/weights/snli_bert_uncased")
parser.add_argument("--generator_dir", type=str, default="bert-base-uncased",
                    help="Path or handle of model to be used as a masked language modeling generator")
parser.add_argument("--max_seq_len", type=int, default=41)
parser.add_argument("--generator_batch_size", type=int, default=8)
parser.add_argument("--model_batch_size", type=int, default=8)

parser.add_argument("--min_samples_per_feature", type=int, default=100,
                    help="Minimum number of samples that get created for each feature for initial variance estimation")
parser.add_argument("--confidence_interval", type=float, default=0.99)
parser.add_argument("--max_abs_error", type=float, default=0.01)

parser.add_argument("--return_generated_samples", action="store_true")
parser.add_argument("--return_model_scores", action="store_true")

parser.add_argument("--experiment_dir", type=str, default=None)
parser.add_argument("--save_every_n_examples", type=int, default=5,
                    help="Save experiment data every N examples in order to avoid losing data on longer computations")


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
    LABEL_TO_IDX = {"entailment": 0, "neutral": 1, "contradiction": 2}
    IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}

    args = parser.parse_args()
    alpha = 1 - args.confidence_interval

    experiment_dir = args.experiment_dir
    if experiment_dir is None:
        experiment_dir = f"snli_compare_required_samples_ci{args.confidence_interval}_maxabserr" \
                         f"{args.max_abs_error:.3f}_minsamplesperfeature{args.min_samples_per_feature}{os.path.sep}"
    args.experiment_dir = experiment_dir

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    with open(os.path.join(experiment_dir, "config.json"), "w") as f_config:
        json.dump(vars(args), f_config, indent=4)

    with open(os.path.join(experiment_dir, "README.txt"), "w") as f_config:
        print(EXPERIMENT_DESCRIPTION, file=f_config)

    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir).to(DEVICE)

    df_train = load_nli(args.train_path).sample(frac=1.0).reset_index(drop=True)
    df_test = load_nli(args.test_path)
    train_set = NLIDataset(premises=df_train["sentence1"].values,
                           hypotheses=df_train["sentence2"].values,
                           labels=df_train["gold_label"].apply(lambda label_str: LABEL_TO_IDX[label_str]).values,
                           tokenizer=tokenizer,
                           max_seq_len=args.max_seq_len)
    test_set = NLIDataset(premises=df_test["sentence1"].values,
                          hypotheses=df_test["sentence2"].values,
                          labels=df_test["gold_label"].apply(lambda label_str: LABEL_TO_IDX[label_str]).values,
                          tokenizer=tokenizer,
                          max_seq_len=args.max_seq_len)

    ime = IMEExplainer(sample_data=train_set.input_ids, return_scores=True, return_num_samples=True,
                       return_samples=args.return_generated_samples, return_variance=True)
    ime_mlm = IMEMaskedLMExplainer(pretrained_name_or_path=args.generator_dir, batch_size=args.generator_batch_size,
                                   return_scores=True, return_num_samples=True,
                                   return_samples=args.return_generated_samples, return_variance=True)

    examples_log = []
    ime_importances = []
    ime_mlm_importances = []
    sequences, labels = [], []
    for idx_example, curr_example in enumerate(DataLoader(test_set, batch_size=1, shuffle=False)):
        @torch.no_grad()
        def _model_wrapper(data: torch.Tensor):
            num_examples = data.shape[0]
            batch_size = min(args.model_batch_size, num_examples)
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


        _curr_example = {k: v.to(DEVICE) for k, v in curr_example.items() if k not in {"labels", "special_tokens_mask"}}
        probas = _model_wrapper(_curr_example["input_ids"])
        predicted_label = int(torch.argmax(probas))
        actual_label = int(curr_example["labels"])

        t1 = time()
        ime_res = ime.explain(curr_example["input_ids"],
                              label=predicted_label,
                              min_samples_per_feature=args.min_samples_per_feature,
                              model_func=_model_wrapper,
                              perturbable_mask=torch.logical_not(curr_example["special_tokens_mask"]))
        t2 = time()

        ime_est_samples = int(estimate_max_samples(ime_res["var"] * args.min_samples_per_feature, alpha=alpha,
                                                   max_abs_error=args.max_abs_error))
        print(f"[IME] Estimated samples required: {ime_est_samples}")
        print(f"[IME] Time taken: {t2 - t1}")

        t3 = time()
        ime_mlm_res = ime_mlm.explain(curr_example["input_ids"],
                                      label=predicted_label,
                                      min_samples_per_feature=args.min_samples_per_feature,
                                      model_func=_model_wrapper,
                                      perturbable_mask=torch.logical_not(curr_example["special_tokens_mask"]))
        t4 = time()

        ime_mlm_est_samples = int(estimate_max_samples(ime_mlm_res["var"] * args.min_samples_per_feature,
                                                       alpha=alpha,
                                                       max_abs_error=args.max_abs_error))
        print(f"[IME MLM] Estimated samples required: {ime_mlm_est_samples}")
        print(f"[IME MLM] Time taken: {t4 - t3}")

        sequence_tokens = tokenizer.convert_ids_to_tokens(curr_example["input_ids"][0])  # type: list

        ime_gen_samples, ime_mlm_gen_samples = [], []
        if args.return_generated_samples:
            for curr_samples in ime_res["samples"]:
                if curr_samples is None:  # non-perturbable feature
                    ime_gen_samples.append([])
                else:
                    ime_gen_samples.append([tokenizer.convert_ids_to_tokens(sample)
                                            for sample in curr_samples.tolist()])

            for curr_samples in ime_mlm_res["samples"]:
                if curr_samples is None:  # non-perturbable feature
                    ime_mlm_gen_samples.append([])
                else:
                    ime_mlm_gen_samples.append([tokenizer.convert_ids_to_tokens(sample)
                                                for sample in curr_samples.tolist()])

        ime_data = {
            "importance": ime_res["importance"].tolist(),
            "var": ime_res["var"].tolist(),
            "num_samples": ime_res["num_samples"].tolist(),
            "samples": ime_gen_samples,
            "scores": [[] if scores is None else scores.tolist()
                       for scores in ime_res["scores"]] if args.return_model_scores else [],
            "est_samples": ime_est_samples,
            "time_taken": t2 - t1
        }

        ime_mlm_data = {
            "importance": ime_mlm_res["importance"].tolist(),
            "var": ime_mlm_res["var"].tolist(),
            "num_samples": ime_mlm_res["num_samples"].tolist(),
            "samples": ime_mlm_gen_samples,
            "scores": [[] if scores is None else scores.tolist()
                       for scores in ime_mlm_res["scores"]] if args.return_model_scores else [],
            "est_samples": ime_mlm_est_samples,
            "time_taken": t4 - t3
        }

        example_data = {
            "sequence": sequence_tokens,
            "predicted_label": IDX_TO_LABEL[predicted_label],
            "actual_label": IDX_TO_LABEL[actual_label],
            "ime_data": ime_data,
            "ime_mlm_data": ime_mlm_data
        }

        examples_log.append(example_data)
        ime_importances.append(ime_res["importance"].tolist())
        ime_mlm_importances.append(ime_mlm_res["importance"].tolist())
        sequences.append(sequence_tokens)
        labels.append(IDX_TO_LABEL[predicted_label])

        if (1 + idx_example) % args.save_every_n_examples == 0:
            print(f"Saving data to {experiment_dir}")

            with open(os.path.join(experiment_dir, "examples.json"), "w") as f:
                json.dump(examples_log, fp=f, indent=4)

            highlight_plot(sequences, labels, ime_importances,
                           path=os.path.join(experiment_dir, "ime_importances.html"))
            highlight_plot(sequences, labels, ime_mlm_importances,
                           path=os.path.join(experiment_dir, "ime_mlm_importances.html"))
