import argparse
import os
from time import time

import torch
from torch.utils.data import DataLoader

from experiments.SNLI.data import load_nli, NLIDataset, LABEL_TO_IDX, IDX_TO_LABEL
from experiments.SNLI.handle_generator import load_generator
from explain_nlp.experimental.core import MethodType, MethodData
from explain_nlp.methods.dependent_ime_mlm import DependentIMEMaskedLMExplainer
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.methods.ime_mlm import IMEMaskedLMExplainer
from explain_nlp.methods.modeling import InterpretableBertForSequenceClassification
from explain_nlp.methods.utils import estimate_max_samples
from explain_nlp.visualizations.highlight import highlight_plot

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="ime", choices=["ime", "ime_mlm", "ime_dependent_mlm"])
parser.add_argument("--min_samples_per_feature", type=int, default=2,
                    help="Minimum number of samples that get created for each feature for initial variance estimation")
parser.add_argument("--confidence_interval", type=float, default=0.99)
parser.add_argument("--max_abs_error", type=float, default=0.01)
parser.add_argument("--return_generated_samples", action="store_true", default=True)
parser.add_argument("--return_model_scores", action="store_true", default=True)

parser.add_argument("--train_path", type=str, default="/home/matej/Documents/data/snli/snli_1.0_train.txt")
parser.add_argument("--test_path", type=str, default="/home/matej/Documents/data/snli/snli_1.0_test_xs.txt")

parser.add_argument("--model_dir", type=str, default="/home/matej/Documents/embeddia/interpretability/ime-lm/examples/weights/snli_bert_uncased")
parser.add_argument("--model_max_seq_len", type=int, default=41)
parser.add_argument("--model_batch_size", type=int, default=2)

parser.add_argument("--generator_type", type=str, default="bert_mlm")
parser.add_argument("--generator_dir", type=str, default="/home/matej/Documents/embeddia/interpretability/ime-lm/examples/weights/bert-base-uncased-snli-mlm",
                    help="Path or handle of model to be used as a language modeling generator")
parser.add_argument("--generator_batch_size", type=int, default=2)
parser.add_argument("--generator_max_seq_len", type=int, default=41)
parser.add_argument("--num_generated_samples", type=int, default=10)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--p_ensure_different", type=float, default=0.0,
                    help="Probability of forcing a generated token to be different from the token in given data")
parser.add_argument("--masked_at_once", type=float, default=None,
                    help="Proportion of tokens to mask out at once during language modeling. By default, mask out one "
                         "token at a time")

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
    masked_at_once = args.masked_at_once if args.masked_at_once is not None else 1
    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")
    print(f"Used device: {DEVICE}")

    experiment_dir = args.experiment_dir
    if experiment_dir is None:
        test_file_name = args.test_path.split(os.path.sep)[-1][:-len(".txt")]  # test file without .txt
        experiment_dir = f"{test_file_name}_compute_required_samples"
    args.experiment_dir = experiment_dir

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Define model and generator
    model = InterpretableBertForSequenceClassification(tokenizer_name=args.model_dir,
                                                       model_name=args.model_dir,
                                                       batch_size=args.model_batch_size,
                                                       max_seq_len=args.model_max_seq_len,
                                                       device="cpu" if args.use_cpu else "cuda")
    temp_model_desc = {"type": "bert", "max_seq_len": args.model_max_seq_len, "handle": args.model_dir}
    generator, gen_desc = load_generator(args)

    # Load sampling data and test data
    df_test = load_nli(args.test_path)
    test_set = NLIDataset(premises=df_test["sentence1"].values,
                          hypotheses=df_test["sentence2"].values,
                          labels=df_test["gold_label"].apply(lambda label_str: LABEL_TO_IDX[label_str]).values,
                          tokenizer=model.tokenizer,
                          max_seq_len=args.model_max_seq_len)

    used_data = {"test_path": args.test_path}
    print(f"Using method '{args.method}'")
    # Define explanation methods
    if args.method == "ime":
        method_type = MethodType.IME
        used_data["train_path"] = args.train_path
        df_train = load_nli(args.train_path).sample(frac=1.0).reset_index(drop=True)
        train_set = NLIDataset(premises=df_train["sentence1"].values,
                               hypotheses=df_train["sentence2"].values,
                               labels=df_train["gold_label"].apply(lambda label_str: LABEL_TO_IDX[label_str]).values,
                               tokenizer=model.tokenizer,
                               max_seq_len=args.model_max_seq_len)
        method = IMEExplainer(sample_data=train_set.input_ids, model=model,
                              return_scores=args.return_model_scores, return_num_samples=True,
                              return_samples=args.return_generated_samples, return_variance=True)
    elif args.method == "ime_mlm":
        method_type = MethodType.INDEPENDENT_IME_MLM
        method = IMEMaskedLMExplainer(model=model, generator=generator,
                                      num_generated_samples=args.num_generated_samples,
                                      return_scores=args.return_model_scores, return_num_samples=True,
                                      return_samples=args.return_generated_samples, return_variance=True)
    elif args.method == "ime_dependent_mlm":
        method_type = MethodType.DEPENDENT_IME_MLM
        method = DependentIMEMaskedLMExplainer(model=model, generator=generator, verbose=args.verbose,
                                               return_scores=args.return_model_scores, return_num_samples=True,
                                               return_samples=args.return_generated_samples, return_variance=True)
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

        _curr_example = {k: v.to(DEVICE) for k, v in curr_example.items() if k not in {"labels", "special_tokens_mask"}}
        probas = model.score(**_curr_example)
        predicted_label = int(torch.argmax(probas, dim=-1))
        actual_label = int(curr_example["labels"])

        t1 = time()
        res = method.explain_text((df_test.iloc[idx_example]["sentence1"],
                                   df_test.iloc[idx_example]["sentence2"]),
                                  label=predicted_label, min_samples_per_feature=args.min_samples_per_feature)
        t2 = time()

        sequence_tokens = res["input"]
        est_samples = int(estimate_max_samples(res["var"] * res["num_samples"],
                                               alpha=alpha, max_abs_error=args.max_abs_error))
        print(f"[{args.method}] Estimated samples required: {est_samples}")
        print(f"[{args.method}] Time taken: {t2 - t1: .3f}s")

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
                                samples=gen_samples, num_estimated_samples=est_samples, time_taken=(t2 - t1),
                                model_scores=[[] if scores is None else scores.tolist()
                                              for scores in res["scores"]] if args.return_model_scores else [])

        if (1 + idx_example) % args.save_every_n_examples == 0:
            print(f"Saving data to {experiment_dir}")
            method_data.save(experiment_dir, file_name=f"{args.method}_data.json")

            highlight_plot(method_data.sequences, method_data.importances,
                           pred_labels=method_data.pred_labels,
                           actual_labels=method_data.actual_labels,
                           path=os.path.join(experiment_dir, f"{args.method}_importances.html"))
