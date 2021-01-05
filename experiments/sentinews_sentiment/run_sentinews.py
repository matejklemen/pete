import os
from time import time

import stanza
import torch
from torch.utils.data import DataLoader, Subset

from explain_nlp.experimental.arguments import parser
from explain_nlp.experimental.core import MethodData
from explain_nlp.experimental.data import load_sentinews, TransformerSeqDataset, LABEL_TO_IDX, IDX_TO_LABEL
from explain_nlp.experimental.handle_explainer import load_explainer
from explain_nlp.experimental.handle_features import handle_features
from explain_nlp.experimental.handle_generator import load_generator
from explain_nlp.methods.modeling import InterpretableBertForSequenceClassification
from explain_nlp.methods.utils import estimate_max_samples
from explain_nlp.visualizations.highlight import highlight_plot

if __name__ == "__main__":
    args = parser.parse_args()

    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")
    print(f"Experiment type: '{args.experiment_type}'")
    print(f"Used device: {DEVICE}")
    print(f"Using method '{args.method}'")
    if args.custom_features is not None:
        print(f"Using custom features: '{args.custom_features}'")

    nlp = stanza.Pipeline(lang="sl", processors="tokenize", use_gpu=not args.use_cpu, tokenize_no_ssplit=True)
    pretokenized_test_data = []
    compute_accurately = args.experiment_type == "accurate_importances"

    if args.experiment_dir is None:
        test_file_name = args.test_path.split(os.path.sep)[-1][:-len(".csv")]  # test file without .txt
        args.experiment_dir = f"{test_file_name}_compute_{args.experiment_type}"

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    # Define model and generator
    model = InterpretableBertForSequenceClassification(tokenizer_name=args.model_dir,
                                                       model_name=args.model_dir,
                                                       batch_size=args.model_batch_size,
                                                       max_seq_len=args.model_max_seq_len,
                                                       max_words=args.model_max_words,
                                                       device="cpu" if args.use_cpu else "cuda")
    model_desc = {"type": "bert", "max_seq_len": args.model_max_seq_len, "handle": args.model_dir}
    generator, gen_desc = load_generator(args,
                                         clm_labels=[IDX_TO_LABEL["sentinews"][i] for i in sorted(IDX_TO_LABEL["sentinews"])])

    df_test = load_sentinews(args.test_path)
    test_set = TransformerSeqDataset(df_test["content"].values,
                                     labels=df_test["sentiment"].apply(
                                         lambda lbl: LABEL_TO_IDX["sentinews"][lbl]).values,
                                     tokenizer=model.tokenizer,
                                     max_seq_len=args.model_max_seq_len)

    if args.method == "whole_word_ime" or args.custom_features is not None:
        pretokenized_test_data = []
        for idx_subset in range((df_test.shape[0] + 1024 - 1) // 1024):
            s, e = idx_subset * 1024, (1 + idx_subset) * 1024
            for s0 in nlp("\n\n".join(df_test["content"].iloc[s: e].values)).sentences:
                pretokenized_test_data.append([token.words[0].text for token in s0.tokens])

    used_data = {"test_path": args.test_path}
    used_sample_data = None
    if args.method in {"ime", "sequential_ime", "whole_word_ime"}:
        used_data["train_path"] = args.train_path
        df_train = load_sentinews(args.train_path).sample(frac=1.0).reset_index(drop=True)
        train_set = TransformerSeqDataset(df_train["content"].values,
                                          labels=df_train["sentiment"].apply(
                                              lambda lbl: LABEL_TO_IDX["sentinews"][lbl]).values,
                                          tokenizer=model.tokenizer, max_seq_len=args.model_max_seq_len)

        used_sample_data = train_set.input_ids
        if args.method == "whole_word_ime":
            pretokenized_train_data = []
            for idx_subset in range((df_train.shape[0] + 1024 - 1) // 1024):
                s, e = idx_subset * 1024, (1 + idx_subset) * 1024
                for s0 in nlp("\n\n".join(df_train["content"].iloc[s: e].values)).sentences:
                    pretokenized_train_data.append([token.words[0].text for token in s0.tokens])

            used_sample_data = model.words_to_internal(pretokenized_train_data)["input_ids"]

    method, method_type = load_explainer(method=args.method, model=model,
                                         confidence_interval=args.confidence_interval if compute_accurately else None,
                                         max_abs_error=args.max_abs_error if compute_accurately else None,
                                         return_model_scores=args.return_model_scores,
                                         return_generated_samples=args.return_generated_samples,
                                         # Method-specific options below:
                                         used_sample_data=used_sample_data, generator=generator,
                                         num_generated_samples=args.num_generated_samples,
                                         controlled=args.controlled,
                                         seed_start_with_ground_truth=args.seed_start_with_ground_truth,
                                         reset_seed_after_first=args.reset_seed_after_first)

    # Container that wraps debugging data and a lot of repetitive appends
    method_data = MethodData(method_type=method_type, model_description=model_desc,
                             generator_description=gen_desc, min_samples_per_feature=args.min_samples_per_feature,
                             possible_labels=[IDX_TO_LABEL["sentinews"][i] for i in sorted(IDX_TO_LABEL["sentinews"])],
                             used_data=used_data, confidence_interval=args.confidence_interval,
                             max_abs_error=args.max_abs_error, custom_features_type=args.custom_features)

    if os.path.exists(os.path.join(args.experiment_dir, f"{args.method}_data.json")):
        method_data = MethodData.load(os.path.join(args.experiment_dir, f"{args.method}_data.json"))

    start_from = args.start_from if args.start_from is not None else len(method_data)
    start_from = min(start_from, len(test_set))
    until = args.until if args.until is not None else len(test_set)
    until = min(until, len(test_set))

    if args.custom_features is not None and args.custom_features.startswith("depparse"):
        nlp = stanza.Pipeline(lang="sl", processors="tokenize,lemma,pos,depparse")

    print(f"Running computation from example#{start_from} (inclusive) to example#{until} (exclusive)")
    for idx_example, curr_example in enumerate(DataLoader(Subset(test_set, range(start_from, until)), batch_size=1, shuffle=False),
                                               start=start_from):
        probas = model.score(**{k: v.to(DEVICE) for k, v in curr_example.items() if k not in {"words",
                                                                                              "labels",
                                                                                              "special_tokens_mask"}})
        predicted_label = int(torch.argmax(probas))
        actual_label = int(curr_example["labels"])

        if args.method == "whole_word_ime":
            input_text = pretokenized_test_data[idx_example]
        else:
            input_text = df_test.iloc[idx_example]["content"]

        curr_features = None
        if args.method in {"ime", "sequential_ime"} and args.custom_features is not None:
            # Obtain word IDs for subwords in all cases as the custom features are usually obtained from words
            encoded = model.to_internal(pretokenized_text_data=[pretokenized_test_data[idx_example]])
            word_ids = encoded["aux_data"]["alignment_ids"][0].tolist()

            curr_features = handle_features(args.custom_features,
                                            word_ids=word_ids,
                                            raw_example=df_test.iloc[idx_example]["content"],
                                            pipe=nlp)
            t1 = time()
            res = method.explain_text(input_text, pretokenized_text_data=pretokenized_test_data[idx_example],
                                      label=predicted_label, min_samples_per_feature=args.min_samples_per_feature,
                                      custom_features=curr_features)
            t2 = time()
        else:
            t1 = time()
            res = method.explain_text(input_text,
                                      label=predicted_label, min_samples_per_feature=args.min_samples_per_feature)
            t2 = time()

        if compute_accurately:
            taken_or_estimated_samples = res['taken_samples']
        else:
            taken_or_estimated_samples = int(estimate_max_samples(res["var"] * res["num_samples"],
                                                                  alpha=(1 - args.confidence_interval),
                                                                  max_abs_error=args.max_abs_error))

        print(f"[{args.method}] {'taken' if compute_accurately else '(estimated) required'} samples: {taken_or_estimated_samples}")
        print(f"[{args.method}] Time taken: {t2 - t1:.2f}s")

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
                                actual_label=actual_label, custom_features=curr_features,
                                importances=res["importance"].tolist(),
                                variances=res["var"].tolist(), num_samples=res["num_samples"].tolist(),
                                samples=gen_samples, num_estimated_samples=res["taken_samples"], time_taken=(t2 - t1),
                                model_scores=[[] if scores is None else scores.tolist()
                                              for scores in res["scores"]] if args.return_model_scores else [])

        if (1 + idx_example) % args.save_every_n_examples == 0:
            print(f"Saving data to {args.experiment_dir}")
            method_data.save(args.experiment_dir, file_name=f"{args.method}_data.json")

            highlight_plot(method_data.sequences, method_data.importances,
                           pred_labels=[method_data.possible_labels[i] for i in method_data.pred_labels],
                           actual_labels=[method_data.possible_labels[i] for i in method_data.actual_labels],
                           custom_features=method_data.custom_features,
                           path=os.path.join(args.experiment_dir, f"{args.method}_importances.html"))
