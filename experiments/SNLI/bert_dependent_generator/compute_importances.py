import argparse
import json
from time import time
import os

import torch
from torch.utils.data import DataLoader

from experiments.SNLI.data import load_nli, NLIDataset, LABEL_TO_IDX, IDX_TO_LABEL
from explain_nlp.methods.dependent_ime_mlm import DependentIMEMaskedLMExplainer
from explain_nlp.methods.generation import BertForMaskedLMGenerator
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.methods.modeling import InterpretableBertForSequenceClassification
from explain_nlp.visualizations.highlight import highlight_plot

EXPERIMENT_DESCRIPTION = \
"""Compute importances with IME/IME+LM. Ran on SNLI. See config.json for specific options."""

parser = argparse.ArgumentParser()

parser.add_argument("--method", type=str, choices=["ime", "ime_mlm"], default="ime_mlm")
parser.add_argument("--train_path", type=str, default="/home/matej/Documents/data/snli/snli_1.0_train.txt",
                    help="Path to data to use for perturbing examples when using IME. ")
parser.add_argument("--test_path", type=str, default="/home/matej/Documents/data/snli/snli_1.0_test_xs.txt")

parser.add_argument("--model_dir", type=str, default="/home/matej/Documents/embeddia/interpretability/ime-lm/examples/weights/snli_bert_uncased")
parser.add_argument("--generator_dir", type=str, default="bert-base-uncased",
                    help="Path or handle of model to be used as a language modeling generator")
parser.add_argument("--generator_max_seq_len", type=int, default=42)
parser.add_argument("--model_max_seq_len", type=int, default=41)
parser.add_argument("--generator_batch_size", type=int, default=2)
parser.add_argument("--model_batch_size", type=int, default=2)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--p_ensure_different", type=float, default=0.0,
                    help="Probability of forcing a generated token to be different from the token in given data")

parser.add_argument("--min_samples_per_feature", type=int, default=2,
                    help="Minimum number of samples that get created for each feature for initial variance estimation")

parser.add_argument("--confidence_interval", type=float, default=0.50)
parser.add_argument("--max_abs_error", type=float, default=1)

parser.add_argument("--return_generated_samples", action="store_true")
parser.add_argument("--return_model_scores", action="store_true")

parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--save_every_n_examples", type=int, default=1,
                    help="Save experiment data every N examples in order to avoid losing data on longer computations")

parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of GPU", default=True)
parser.add_argument("--verbose", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    alpha = 1 - args.confidence_interval
    masked_at_once = 1
    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")
    print(f"Used device: {DEVICE}")

    experiment_dir = args.experiment_dir
    if experiment_dir is None:
        experiment_dir = f"snli_importances_{args.method}_ci{args.confidence_interval}_maxabserr" \
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
                                                       max_seq_len=args.model_max_seq_len,
                                                       device="cpu" if args.use_cpu else "cuda")

    df_test = load_nli(args.test_path)
    test_set = NLIDataset(premises=df_test["sentence1"].values,
                          hypotheses=df_test["sentence2"].values,
                          labels=df_test["gold_label"].apply(lambda label_str: LABEL_TO_IDX[label_str]).values,
                          tokenizer=model.tokenizer,
                          max_seq_len=args.model_max_seq_len)

    print(f"Using method '{args.method}'")
    if args.method == "ime":
        df_train = load_nli(args.train_path).sample(frac=1.0).reset_index(drop=True)
        train_set = NLIDataset(premises=df_train["sentence1"].values,
                               hypotheses=df_train["sentence2"].values,
                               labels=df_train["gold_label"].apply(lambda label_str: LABEL_TO_IDX[label_str]).values,
                               tokenizer=model.tokenizer,
                               max_seq_len=args.model_max_seq_len)
        explainer = IMEExplainer(sample_data=train_set.input_ids, model=model,
                                 confidence_interval=args.confidence_interval, max_abs_error=args.max_abs_error,
                                 return_scores=args.return_model_scores, return_num_samples=True,
                                 return_samples=args.return_generated_samples, return_variance=True)
    else:
        generator = BertForMaskedLMGenerator(tokenizer_name=args.generator_dir,
                                             model_name=args.generator_dir,
                                             batch_size=args.generator_batch_size,
                                             max_seq_len=args.generator_max_seq_len,
                                             device="cpu" if args.use_cpu else "cuda",
                                             top_p=args.top_p,
                                             masked_at_once=1,
                                             p_ensure_different=args.p_ensure_different)
        explainer = DependentIMEMaskedLMExplainer(model=model, generator=generator, verbose=args.verbose,
                                                  return_scores=args.return_model_scores, return_num_samples=True,
                                                  return_samples=args.return_generated_samples, return_variance=True)

    examples_log = []
    importances = []
    sequences, labels = [], []

    # If directory exists, try loading existing data
    if os.path.exists(os.path.join(experiment_dir, "examples.json")):
        with open(os.path.join(experiment_dir, "examples.json")) as f:
            examples_log = json.load(f)
        importances = [ex[f"{args.method}_data"]["importance"] for ex in examples_log]
        sequences = [ex["sequence"] for ex in examples_log]
        labels = [ex["predicted_label"] for ex in examples_log]

        print(f"Loaded data for {len(examples_log)} existing examples!")

    start_from = len(examples_log)
    print(f"Starting from example#{start_from}")
    for idx_example, curr_example in enumerate(DataLoader(test_set, batch_size=1, shuffle=False)):
        if idx_example < start_from:
            continue

        _curr_example = {k: v.to(DEVICE) for k, v in curr_example.items() if k not in {"labels", "special_tokens_mask"}}
        probas = model.score(**_curr_example)
        predicted_label = int(torch.argmax(probas))
        actual_label = int(curr_example["labels"])

        t1 = time()
        res = explainer.explain_text((df_test.iloc[idx_example]["sentence1"],
                                      df_test.iloc[idx_example]["sentence2"]),
                                     label=predicted_label,
                                     min_samples_per_feature=args.min_samples_per_feature)
        t2 = time()
        print(f"[{args.method}] Taken samples: {res['taken_samples']}")
        print(f"[{args.method}] Time taken: {t2 - t1}")

        sequence_tokens = res["input"]

        gen_samples = []
        if args.return_generated_samples:
            for curr_samples in res["samples"]:
                if curr_samples is None:  # non-perturbable feature
                    gen_samples.append([])
                else:
                    gen_samples.append(explainer.model.convert_ids_to_tokens(curr_samples))

        explainer_data = {
            "importance": res["importance"].tolist(),
            "var": res["var"].tolist(),
            "num_samples": res["num_samples"].tolist(),
            "samples": gen_samples,
            "scores": [[] if scores is None else scores.tolist()
                       for scores in res["scores"]] if args.return_model_scores else [],
            "est_samples": res["taken_samples"],
            "time_taken": t2 - t1
        }

        example_data = {
            "sequence": sequence_tokens,
            "predicted_label": IDX_TO_LABEL[predicted_label],
            "actual_label": IDX_TO_LABEL[actual_label],
            "ime_data": [],
            "ime_mlm_data": []
        }
        example_data[f"{args.method}_data"] = explainer_data

        examples_log.append(example_data)
        importances.append(res["importance"].tolist())
        sequences.append(sequence_tokens)
        labels.append(IDX_TO_LABEL[predicted_label])

        if (1 + idx_example) % args.save_every_n_examples == 0:
            print(f"Saving data to {experiment_dir}")

            with open(os.path.join(experiment_dir, "examples.json"), "w") as f:
                json.dump(examples_log, fp=f, indent=4)

            highlight_plot(sequences, labels, importances,
                           path=os.path.join(experiment_dir, f"{args.method}_importances.html"))
