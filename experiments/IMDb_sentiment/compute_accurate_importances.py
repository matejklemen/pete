import argparse
import os
from time import time

import stanza
import torch
from torch.utils.data import DataLoader

from experiments.IMDb_sentiment.data import load_imdb, SequenceDataset, IDX_TO_LABEL
from experiments.IMDb_sentiment.handle_generator import load_generator
from explain_nlp.experimental.core import MethodData, MethodType
from explain_nlp.methods.dependent_ime_mlm import DependentIMEMaskedLMExplainer
from explain_nlp.methods.ime import IMEExplainer, SequentialIMEExplainer, WholeWordIMEExplainer
from explain_nlp.methods.ime_mlm import IMEMaskedLMExplainer
from explain_nlp.methods.modeling import InterpretableBertForSequenceClassification
from explain_nlp.visualizations.highlight import highlight_plot


parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="whole_word_ime", choices=["ime", "sequential_ime", "whole_word_ime",
                                                                             "ime_mlm", "ime_dependent_mlm"])
parser.add_argument("--min_samples_per_feature", type=int, default=2,
                    help="Minimum number of samples that get created for each feature for initial variance estimation")
parser.add_argument("--confidence_interval", type=float, default=0.5)
parser.add_argument("--max_abs_error", type=float, default=1.00)
parser.add_argument("--return_generated_samples", action="store_true")
parser.add_argument("--return_model_scores", action="store_true")

parser.add_argument("--train_path", type=str, default="/home/matej/Documents/data/aclImdb/train/data.csv")
parser.add_argument("--test_path", type=str, default="/home/matej/Documents/data/aclImdb/test/data_xs.csv")

parser.add_argument("--model_dir", type=str, default="/home/matej/Documents/embeddia/interpretability/ime-lm/examples/weights/imdb_model")
parser.add_argument("--model_max_seq_len", type=int, default=128)
parser.add_argument("--model_max_words", type=int, default=120)  # Actually,  95th percentile: 695 (too long; simplify)
parser.add_argument("--model_batch_size", type=int, default=2)

parser.add_argument("--generator_type", type=str, default="bert_mlm")
parser.add_argument("--controlled", action="store_true",
                    help="Whether to use controlled LM/MLM for generation")
parser.add_argument("--generator_dir", type=str, default="bert-base-uncased",
                    help="Path or handle of model to be used as a language modeling generator")
parser.add_argument("--generator_batch_size", type=int, default=2)
parser.add_argument("--generator_max_seq_len", type=int, default=128)
parser.add_argument("--num_generated_samples", type=int, default=10)
parser.add_argument("--top_p", type=float, default=None)

# Experimental (only in Bert MLM generator) for now
parser.add_argument("--strategy", type=str, choices=["top_k", "top_p", "threshold", "num_samples"], default="top_k")
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--threshold", type=float, default=0.1)

parser.add_argument("--seed_start_with_ground_truth", action="store_true")
parser.add_argument("--reset_seed_after_first", action="store_true")

parser.add_argument("--experiment_dir", type=str, default=None)
parser.add_argument("--save_every_n_examples", type=int, default=1,
                    help="Save experiment data every N examples in order to avoid losing data on longer computations")

parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of GPU")
parser.add_argument("--verbose", action="store_true")

parser.add_argument("--start_from", type=int, default=None, help="From which example onwards to do computation")
parser.add_argument("--until", type=int, default=None, help="Until which example to do computation")


if __name__ == "__main__":
    args = parser.parse_args()

    alpha = 1 - args.confidence_interval
    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")
    print(f"Used device: {DEVICE}")
    nlp = stanza.Pipeline(lang="en", processors="tokenize", use_gpu=not args.use_cpu, tokenize_no_ssplit=True)
    pretokenized_test_data = []

    experiment_dir = args.experiment_dir
    if experiment_dir is None:
        test_file_name = args.test_path.split(os.path.sep)[-1][:-len(".csv")]  # test file without .txt
        experiment_dir = f"{test_file_name}_compute_accurate_importances"
    args.experiment_dir = experiment_dir

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Define model and generator
    model = InterpretableBertForSequenceClassification(tokenizer_name=args.model_dir,
                                                       model_name=args.model_dir,
                                                       batch_size=args.model_batch_size,
                                                       max_seq_len=args.model_max_seq_len,
                                                       max_words=args.model_max_words,
                                                       device="cpu" if args.use_cpu else "cuda")
    temp_model_desc = {"type": "bert", "max_seq_len": args.model_max_seq_len, "handle": args.model_dir}
    generator, gen_desc = load_generator(args)

    df_test = load_imdb(args.test_path)
    test_set = SequenceDataset(reviews=df_test["review"].values,
                               labels=df_test["label"].values,
                               tokenizer=model.tokenizer,
                               max_seq_len=args.model_max_seq_len)
    if args.method == "whole_word_ime":
        pretokenized_test_data = []
        for idx_subset in range((df_test.shape[0] + 1024 - 1) // 1024):
            s, e = idx_subset * 1024, (1 + idx_subset) * 1024
            for s0 in nlp("\n\n".join(df_test["review"].iloc[s: e].values)).sentences:
                pretokenized_test_data.append([token.words[0].text for token in s0.tokens])

    used_data = {"test_path": args.test_path}
    print(f"Using method '{args.method}'")
    # Define explanation methods
    if args.method in {"ime", "sequential_ime", "whole_word_ime"}:
        method_type = MethodType.IME
        used_data["train_path"] = args.train_path
        df_train = load_imdb(args.train_path).sample(frac=1.0).reset_index(drop=True)
        train_set = SequenceDataset(reviews=df_train["review"].values,
                                    labels=df_train["label"].values,
                                    tokenizer=model.tokenizer,
                                    max_seq_len=args.model_max_seq_len)
        explainer_cls = IMEExplainer if args.method == "ime" else SequentialIMEExplainer

        used_sample_data = train_set.input_ids
        if args.method == "whole_word_ime":
            explainer_cls = WholeWordIMEExplainer
            print(f"Tokenizing train data ({df_train.shape[0]} examples)")

            pretokenized_train_data = []
            for idx_subset in range((df_train.shape[0] + 1024 - 1) // 1024):
                s, e = idx_subset * 1024, (1 + idx_subset) * 1024
                for s0 in nlp("\n\n".join(df_train["review"].iloc[s: e].values)).sentences:
                    pretokenized_train_data.append([token.words[0].text for token in s0.tokens])

            used_sample_data = model.words_to_internal(pretokenized_train_data)["input_ids"]

        method = explainer_cls(sample_data=used_sample_data, model=model,
                               confidence_interval=args.confidence_interval, max_abs_error=args.max_abs_error,
                               return_scores=args.return_model_scores, return_num_samples=True,
                               return_samples=args.return_generated_samples, return_variance=True)
    elif args.method == "ime_mlm":
        method_type = MethodType.INDEPENDENT_IME_MLM
        method = IMEMaskedLMExplainer(model=model, generator=generator,
                                      confidence_interval=args.confidence_interval, max_abs_error=args.max_abs_error,
                                      num_generated_samples=args.num_generated_samples,
                                      return_scores=args.return_model_scores, return_num_samples=True,
                                      return_samples=args.return_generated_samples, return_variance=True)
    elif args.method == "ime_dependent_mlm":
        method_type = MethodType.DEPENDENT_IME_MLM
        method = DependentIMEMaskedLMExplainer(model=model, generator=generator, verbose=args.verbose,
                                               confidence_interval=args.confidence_interval, max_abs_error=args.max_abs_error,
                                               return_scores=args.return_model_scores, return_num_samples=True,
                                               return_samples=args.return_generated_samples, return_variance=True,
                                               controlled=args.controlled,
                                               seed_start_with_ground_truth=args.seed_start_with_ground_truth,
                                               reset_seed_after_first=args.reset_seed_after_first)
    else:
        raise NotImplementedError(f"Unsupported method: '{args.method}'")

    # Container that wraps debugging data and a lot of repetitive appends
    method_data = MethodData(method_type=method_type, model_description=temp_model_desc,
                             generator_description=gen_desc, min_samples_per_feature=args.min_samples_per_feature,
                             possible_labels=[IDX_TO_LABEL[i] for i in sorted(IDX_TO_LABEL.keys())],
                             used_data=used_data, confidence_interval=args.confidence_interval,
                             max_abs_error=args.max_abs_error)

    if os.path.exists(os.path.join(experiment_dir, f"{args.method}_data.json")):
        method_data = MethodData.load(os.path.join(experiment_dir, f"{args.method}_data.json"))

    start_from = args.start_from if args.start_from is not None else len(method_data)
    start_from = min(start_from, len(test_set))
    until = args.until if args.until is not None else len(test_set)
    until = min(until, len(test_set))

    print(f"Running computation from example#{start_from} (inclusive) to example#{until} (exclusive)")
    for idx_example, curr_example in enumerate(DataLoader(test_set, batch_size=1, shuffle=False)):
        if idx_example < start_from:
            continue
        if idx_example >= until:
            break

        _curr_example = {k: v.to(DEVICE) for k, v in curr_example.items() if k not in {"words",
                                                                                       "labels",
                                                                                       "special_tokens_mask"}}
        probas = model.score(**_curr_example)
        predicted_label = int(torch.argmax(probas))
        actual_label = int(curr_example["labels"])

        input_text = df_test.iloc[idx_example]["review"]
        if args.method == "whole_word_ime":
            input_text = pretokenized_test_data[idx_example]

        t1 = time()
        res = method.explain_text(input_text,
                                  label=predicted_label, min_samples_per_feature=args.min_samples_per_feature)
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
                    gen_samples.append(method.model.from_internal(curr_samples, skip_special_tokens=False,
                                                                  take_as_single_sequence=True))

        method_data.add_example(sequence=sequence_tokens, label=predicted_label, probas=probas[0].tolist(),
                                actual_label=actual_label, importances=res["importance"].tolist(),
                                variances=res["var"].tolist(), num_samples=res["num_samples"].tolist(),
                                samples=gen_samples, num_estimated_samples=res["taken_samples"], time_taken=(t2 - t1),
                                model_scores=[[] if scores is None else scores.tolist()
                                              for scores in res["scores"]] if args.return_model_scores else [])

        if (1 + idx_example) % args.save_every_n_examples == 0:
            print(f"Saving data to {experiment_dir}")
            method_data.save(experiment_dir, file_name=f"{args.method}_data.json")

            highlight_plot(method_data.sequences, method_data.importances,
                           pred_labels=[method_data.possible_labels[i] for i in method_data.pred_labels],
                           actual_labels=[method_data.possible_labels[i] for i in method_data.actual_labels],
                           path=os.path.join(experiment_dir, f"{args.method}_importances.html"))