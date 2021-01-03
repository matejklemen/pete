import os
from time import time

import stanza
import torch
from torch.utils.data import DataLoader, Subset

from experiments.SNLI.data import load_nli, NLIDataset, LABEL_TO_IDX, IDX_TO_LABEL
from experiments.SNLI.handle_generator import load_generator
from explain_nlp.experimental.core import MethodData, MethodType
from explain_nlp.experimental.arguments import parser
from explain_nlp.methods.dependent_ime_mlm import DependentIMEMaskedLMExplainer
from explain_nlp.methods.features import extract_groups, stanza_word_features, depparse_custom_groups_1
from explain_nlp.methods.ime import IMEExplainer, SequentialIMEExplainer, WholeWordIMEExplainer
from explain_nlp.methods.ime_mlm import IMEMaskedLMExplainer
from explain_nlp.methods.modeling import InterpretableBertForSequenceClassification
from explain_nlp.visualizations.highlight import highlight_plot

if __name__ == "__main__":
    args = parser.parse_args()

    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")
    print(f"Used device: {DEVICE}")
    nlp = stanza.Pipeline(lang="en", processors="tokenize", use_gpu=not args.use_cpu, tokenize_no_ssplit=True)
    pretokenized_test_data = []

    if args.experiment_dir is None:
        test_file_name = args.test_path.split(os.path.sep)[-1][:-len(".txt")]  # test file without .txt
        args.experiment_dir = f"{test_file_name}_compute_accurate_importances"

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    # Define model and generator
    model = InterpretableBertForSequenceClassification(tokenizer_name=args.model_dir,
                                                       model_name=args.model_dir,
                                                       batch_size=args.model_batch_size,
                                                       max_seq_len=args.model_max_seq_len,
                                                       max_words=args.model_max_words,
                                                       device="cpu" if args.use_cpu else "cuda")
    temp_model_desc = {"type": "bert", "max_seq_len": args.model_max_seq_len, "handle": args.model_dir}
    generator, gen_desc = load_generator(args)

    df_test = load_nli(args.test_path)
    test_set = NLIDataset(premises=df_test["sentence1"].values,
                          hypotheses=df_test["sentence2"].values,
                          labels=df_test["gold_label"].apply(lambda label_str: LABEL_TO_IDX[label_str]).values,
                          tokenizer=model.tokenizer,
                          max_seq_len=args.model_max_seq_len)

    if args.method == "whole_word_ime" or args.custom_features is not None:
        pretokenized_test_data = []
        for idx_subset in range((df_test.shape[0] + 1024 - 1) // 1024):
            s, e = idx_subset * 1024, (1 + idx_subset) * 1024
            for s0, s1 in zip(nlp("\n\n".join(df_test["sentence1"].iloc[s: e].values)).sentences,
                              nlp("\n\n".join(df_test["sentence2"].iloc[s: e].values)).sentences):
                pretokenized_test_data.append((
                    [token.words[0].text for token in s0.tokens],
                    [token.words[0].text for token in s1.tokens]
                ))

    used_data = {"test_path": args.test_path}
    print(f"Using method '{args.method}'")
    if args.custom_features is not None:
        print(f"Using custom features: '{args.custom_features}'")

    # Define explanation methods
    if args.method in {"ime", "sequential_ime", "whole_word_ime"}:
        method_type = MethodType.IME
        used_data["train_path"] = args.train_path
        df_train = load_nli(args.train_path).sample(frac=1.0).reset_index(drop=True)
        train_set = NLIDataset(premises=df_train["sentence1"].values,
                               hypotheses=df_train["sentence2"].values,
                               labels=df_train["gold_label"].apply(lambda label_str: LABEL_TO_IDX[label_str]).values,
                               tokenizer=model.tokenizer,
                               max_seq_len=args.model_max_seq_len)
        explainer_cls = IMEExplainer if args.method == "ime" else SequentialIMEExplainer

        used_sample_data = train_set.input_ids
        if args.method == "whole_word_ime":
            explainer_cls = WholeWordIMEExplainer

            pretokenized_train_data = []
            for idx_subset in range((df_train.shape[0] + 1024 - 1) // 1024):
                s, e = idx_subset * 1024, (1 + idx_subset) * 1024
                for s0, s1 in zip(nlp("\n\n".join(df_train["sentence1"].iloc[s: e].values)).sentences,
                                  nlp("\n\n".join(df_train["sentence2"].iloc[s: e].values)).sentences):
                    pretokenized_train_data.append((
                        [token.words[0].text for token in s0.tokens],
                        [token.words[0].text for token in s1.tokens]
                    ))

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
                             max_abs_error=args.max_abs_error, custom_features_type=args.custom_features)

    if os.path.exists(os.path.join(args.experiment_dir, f"{args.method}_data.json")):
        method_data = MethodData.load(os.path.join(args.experiment_dir, f"{args.method}_data.json"))

    start_from = args.start_from if args.start_from is not None else len(method_data)
    start_from = min(start_from, len(test_set))
    until = args.until if args.until is not None else len(test_set)
    until = min(until, len(test_set))

    if args.custom_features is not None and args.custom_features.startswith("depparse"):
        nlp = stanza.Pipeline(lang="en", processors="tokenize,lemma,pos,depparse")

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
            input_text = (df_test.iloc[idx_example]["sentence1"], df_test.iloc[idx_example]["sentence2"])

        curr_features = None
        if args.method in {"ime", "sequential_ime"} and args.custom_features is not None:
            # Obtain word IDs for subwords in all cases as the custom features are usually obtained from words
            encoded = model.to_internal(pretokenized_text_data=[pretokenized_test_data[idx_example]])
            word_ids = encoded["aux_data"]["alignment_ids"][0].tolist()

            if args.custom_features == "words":
                feature_ids = word_ids
            elif args.custom_features == "sentences":
                res = stanza_word_features(
                    raw_example=(df_test.iloc[idx_example]["sentence1"], df_test.iloc[idx_example]["sentence2"]),
                    pipe=nlp
                )
                feature_ids = [res["word_id_to_sent_id"].get(curr_word_id, -1) for curr_word_id in word_ids]
            elif args.custom_features.startswith("depparse"):
                res = stanza_word_features(
                    raw_example=(df_test.iloc[idx_example]["sentence1"], df_test.iloc[idx_example]["sentence2"]),
                    pipe=nlp,
                    do_depparse=True
                )

                if args.custom_features == "depparse_simple":
                    custom_groups = depparse_custom_groups_1(res["word_id_to_head_id"], res["word_id_to_deprel"])
                else:
                    raise ValueError(f"Unrecognized option for custom_features: '{args.custom_features}'")

                feature_ids = [custom_groups.get(curr_word_id, -1) for curr_word_id in word_ids]
            else:
                raise NotImplementedError

            input_tokens = model.tokenizer.convert_ids_to_tokens(curr_example["input_ids"][0])
            curr_features = extract_groups(feature_ids, ignore_index=-1)

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
