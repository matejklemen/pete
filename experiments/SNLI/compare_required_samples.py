import argparse
import json
import os
from time import time

import torch
from torch.utils.data import DataLoader

from explain_nlp.methods.generation import BertForMaskedLMGenerator
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.methods.ime_mlm import IMEMaskedLMExplainer
from explain_nlp.methods.modeling import InterpretableBertForSequenceClassification
from explain_nlp.methods.utils import estimate_max_samples
from explain_nlp.visualizations.highlight import highlight_plot

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
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--masked_at_once", type=float, default=None,
                    help="Proportion of tokens to mask out at once during masked language modeling. By default, "
                         "mask out one token at a time")
parser.add_argument("--p_ensure_different", type=float, default=0.0,
                    help="Probability of forcing a generated token to be different from the token in given data")

parser.add_argument("--num_generated_samples", type=int, default=100)
parser.add_argument("--min_samples_per_feature", type=int, default=10,
                    help="Minimum number of samples that get created for each feature for initial variance estimation")
parser.add_argument("--confidence_interval", type=float, default=0.99)
parser.add_argument("--max_abs_error", type=float, default=0.01)

parser.add_argument("--return_generated_samples", action="store_true")
parser.add_argument("--return_model_scores", action="store_true")

parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--save_every_n_examples", type=int, default=5,
                    help="Save experiment data every N examples in order to avoid losing data on longer computations")

parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of GPU")


if __name__ == "__main__":
    LABEL_TO_IDX = {"entailment": 0, "neutral": 1, "contradiction": 2}
    IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}

    args = parser.parse_args()
    alpha = 1 - args.confidence_interval
    masked_at_once = args.masked_at_once if args.masked_at_once is not None else 1
    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")

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

    model = InterpretableBertForSequenceClassification(tokenizer_name=args.model_dir,
                                                       model_name=args.model_dir,
                                                       batch_size=args.model_batch_size,
                                                       max_seq_len=args.max_seq_len,
                                                       device="cpu" if args.use_cpu else "cuda")
    generator = BertForMaskedLMGenerator(tokenizer_name=args.generator_dir,
                                         model_name=args.generator_dir,
                                         batch_size=args.generator_batch_size,
                                         max_seq_len=args.max_seq_len,
                                         device="cpu" if args.use_cpu else "cuda",
                                         top_p=args.top_p,
                                         masked_at_once=masked_at_once,
                                         p_ensure_different=args.p_ensure_different)

    df_train = load_nli(args.train_path).sample(frac=1.0).reset_index(drop=True)
    df_test = load_nli(args.test_path)
    train_set = NLIDataset(premises=df_train["sentence1"].values,
                           hypotheses=df_train["sentence2"].values,
                           labels=df_train["gold_label"].apply(lambda label_str: LABEL_TO_IDX[label_str]).values,
                           tokenizer=model.tokenizer,
                           max_seq_len=args.max_seq_len)
    test_set = NLIDataset(premises=df_test["sentence1"].values,
                          hypotheses=df_test["sentence2"].values,
                          labels=df_test["gold_label"].apply(lambda label_str: LABEL_TO_IDX[label_str]).values,
                          tokenizer=model.tokenizer,
                          max_seq_len=args.max_seq_len)

    ime = IMEExplainer(sample_data=train_set.input_ids, model=model,
                       return_scores=args.return_model_scores, return_num_samples=True,
                       return_samples=args.return_generated_samples, return_variance=True)
    ime_mlm = IMEMaskedLMExplainer(model=model, generator=generator, num_generated_samples=args.num_generated_samples,
                                   return_scores=args.return_model_scores, return_num_samples=True,
                                   return_samples=args.return_generated_samples, return_variance=True)

    examples_log = []
    ime_importances, ime_mlm_importances = [], []
    sequences, labels = [], []

    # If directory exists, try loading existing data
    if os.path.exists(os.path.join(experiment_dir, "examples.json")):
        with open(os.path.join(experiment_dir, "examples.json")) as f:
            examples_log = json.load(f)
        ime_importances = [ex["ime_data"]["importance"] for ex in examples_log]
        ime_mlm_importances = [ex["ime_mlm_data"]["importance"] for ex in examples_log]
        sequences = [ex["sequence"] for ex in examples_log]
        labels = [ex["predicted_label"] for ex in examples_log]

        print(f"Loaded data for {len(examples_log)} existing examples!")

    for idx_example, curr_example in enumerate(DataLoader(test_set, batch_size=1, shuffle=False)):
        _curr_example = {k: v.to(DEVICE) for k, v in curr_example.items() if k not in {"labels", "special_tokens_mask"}}
        probas = model.score(**_curr_example)
        predicted_label = int(torch.argmax(probas, dim=-1))
        actual_label = int(curr_example["labels"])

        t1 = time()
        ime_res = ime.explain_text((df_test.iloc[idx_example]["sentence1"],
                                    df_test.iloc[idx_example]["sentence2"]),
                                   label=predicted_label,
                                   min_samples_per_feature=args.min_samples_per_feature)
        t2 = time()

        ime_est_samples = int(estimate_max_samples(ime_res["var"] * ime_res["num_samples"],
                                                   alpha=alpha, max_abs_error=args.max_abs_error))
        print(f"[IME] Estimated samples required: {ime_est_samples}")
        print(f"[IME] Time taken: {t2 - t1: .3f}s")

        t3 = time()
        ime_mlm_res = ime_mlm.explain_text((df_test.iloc[idx_example]["sentence1"],
                                            df_test.iloc[idx_example]["sentence2"]),
                                           label=predicted_label,
                                           min_samples_per_feature=args.min_samples_per_feature)
        t4 = time()

        ime_mlm_est_samples = int(estimate_max_samples(ime_mlm_res["var"] * ime_mlm_res["num_samples"],
                                                       alpha=alpha, max_abs_error=args.max_abs_error))
        print(f"[IME MLM] Estimated samples required: {ime_mlm_est_samples}")
        print(f"[IME MLM] Time taken: {t4 - t3: .3f}s")

        sequence_tokens = ime_mlm_res["input"]

        ime_gen_samples, ime_mlm_gen_samples = [], []
        if args.return_generated_samples:
            for curr_samples in ime_res["samples"]:
                if curr_samples is None:  # non-perturbable feature
                    ime_gen_samples.append([])
                else:
                    ime_gen_samples.append(ime.model.convert_ids_to_tokens(curr_samples))

            for curr_samples in ime_mlm_res["samples"]:
                if curr_samples is None:  # non-perturbable feature
                    ime_mlm_gen_samples.append([])
                else:
                    ime_mlm_gen_samples.append(ime_mlm.model.convert_ids_to_tokens(curr_samples))

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
