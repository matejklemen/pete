import logging
import os
import sys
from time import time

import stanza
import torch

from explain_nlp.experimental.arguments import methods_parser, runtime_parse_args
from explain_nlp.experimental.core import MethodData
from explain_nlp.experimental.data import load_nli, TransformerSeqPairDataset, LABEL_TO_IDX, IDX_TO_LABEL
from explain_nlp.experimental.handle_explainer import load_explainer
from explain_nlp.experimental.handle_features import handle_features
from explain_nlp.experimental.handle_generator import load_generator
from explain_nlp.methods.hybrid import create_uniform_weights
from explain_nlp.methods.utils import estimate_feature_samples
from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification
from explain_nlp.visualizations.highlight import highlight_plot

if __name__ == "__main__":
    args = methods_parser.parse_args(["ime"])
    args = runtime_parse_args(args)
    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")
    compute_accurately = args.method_class == "ime" and args.experiment_type == "accurate_importances"

    # Set up logging to file and stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout),
                         logging.FileHandler(os.path.join(args.experiment_dir, "experiment.log"))]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)

    nlp = stanza.Pipeline(lang="en", processors="tokenize", use_gpu=not args.use_cpu, tokenize_no_ssplit=True)
    pretokenized_test_data = None

    model = InterpretableBertForSequenceClassification(tokenizer_name=args.model_dir,
                                                       model_name=args.model_dir,
                                                       batch_size=args.model_batch_size,
                                                       max_seq_len=args.model_max_seq_len,
                                                       max_words=args.model_max_words,
                                                       device="cpu" if args.use_cpu else "cuda")
    model_description = {"type": "bert", "max_seq_len": args.model_max_seq_len, "handle": args.model_dir}
    generator, gen_description = load_generator(args,
                                                clm_labels=["<ENTAILMENT>", "<NEUTRAL>", "<CONTRADICTION>"])

    df_test = load_nli(args.test_path)
    if args.custom_features is not None:
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
    used_sample_data = None
    data_weights = None
    if args.method in {"ime", "ime_hybrid"}:
        logging.info("Loading training dataset as sampling data")
        used_data["train_path"] = args.train_path
        df_train = load_nli(args.train_path).sample(frac=1.0).reset_index(drop=True)
        used_tokenizer = model.tokenizer if args.method == "ime" else generator.tokenizer
        train_set = TransformerSeqPairDataset.build(df_train["sentence1"].tolist(), df_train["sentence2"].tolist(),
                                                    labels=df_train["gold_label"].apply(
                                                        lambda label_str: LABEL_TO_IDX["snli"][label_str]).tolist(),
                                                    tokenizer=used_tokenizer, max_seq_len=args.model_max_seq_len)
        data_weights = create_uniform_weights(train_set.input_ids, train_set.special_tokens_masks)
        used_sample_data = train_set.input_ids

    method, method_type = load_explainer(model=model, generator=generator,
                                         used_sample_data=used_sample_data, data_weights=data_weights,
                                         **vars(args))
    num_taken_samples = args.min_samples_per_feature if args.method_class == "ime" else args.num_samples

    # Container that wraps debugging data and a lot of repetitive appends
    method_data = MethodData(method_type=method_type,
                             model_description=model_description, generator_description=gen_description,
                             possible_labels=[IDX_TO_LABEL["snli"][i] for i in sorted(IDX_TO_LABEL["snli"])],
                             min_samples_per_feature=
                             (args.min_samples_per_feature if args.method_class == "ime" else args.num_samples),
                             confidence_interval=
                             (args.confidence_interval if args.method_class == "ime" else None),
                             max_abs_error=
                             (args.max_abs_error if args.method_class == "ime" else None),
                             used_data=used_data, custom_features_type=args.custom_features)

    # Load existing data for experiment and make sure the computation is not needlessly reran
    if os.path.exists(os.path.join(args.experiment_dir, f"{args.method}_data.json")):
        method_data = MethodData.load(os.path.join(args.experiment_dir, f"{args.method}_data.json"))

    start_from = args.start_from if args.start_from is not None else len(method_data)
    start_from = min(start_from, df_test.shape[0])
    until = args.until if args.until is not None else df_test.shape[0]
    until = min(until, df_test.shape[0])

    if args.custom_features is not None:
        # Reload pipeline if depparse features are used (not done from the start as this would slow down tokenization)
        if args.custom_features.startswith("depparse"):
            nlp = stanza.Pipeline(lang="en", processors="tokenize,lemma,pos,depparse")
        else:
            nlp = stanza.Pipeline(lang="en", processors="tokenize")

    logging.info(f"Running computation from example#{start_from} (inclusive) to example#{until} (exclusive)")
    for idx_example, input_pair in enumerate(df_test.iloc[start_from: until][["sentence1", "sentence2"]].values.tolist(),
                                             start=start_from):
        if args.custom_features is not None:
            encoded_example = model.to_internal(pretokenized_text_data=[pretokenized_test_data[idx_example]])
        else:
            encoded_example = model.to_internal(text_data=[input_pair])

        probas = model.score(input_ids=encoded_example["input_ids"].to(DEVICE),
                             **{k: v.to(DEVICE) for k, v in encoded_example["aux_data"].items()})
        predicted_label = int(torch.argmax(probas))
        actual_label = int(df_test.iloc[[idx_example]]["gold_label"].apply(
                                                        lambda label_str: LABEL_TO_IDX["snli"][label_str]))

        pretokenized_example, curr_features = None, None
        if args.custom_features is not None:
            # Obtain word IDs for subwords in all cases as the custom features are usually obtained from words
            word_ids = encoded_example["aux_data"]["alignment_ids"][0].tolist()
            curr_features = handle_features(args.custom_features,
                                            word_ids=word_ids,
                                            raw_example=(df_test.iloc[idx_example]["sentence1"],
                                                         df_test.iloc[idx_example]["sentence2"]),
                                            pipe=nlp)
            pretokenized_example = pretokenized_test_data[idx_example]

        side_results = {}
        t1 = time()
        if args.method_class == "lime":
            res = method.explain_text(input_pair, pretokenized_text_data=pretokenized_example,
                                      label=predicted_label, num_samples=num_taken_samples,
                                      explanation_length=args.explanation_length,
                                      custom_features=curr_features)
            t2 = time()
        else:
            res = method.explain_text(input_pair, pretokenized_text_data=pretokenized_example,
                                      label=predicted_label, min_samples_per_feature=num_taken_samples,
                                      custom_features=curr_features)
            t2 = time()

            side_results["variances"] = res["var"].tolist()
            side_results["num_samples"] = res["num_samples"].tolist()

            if compute_accurately:
                taken_or_estimated_samples = res['taken_samples']
            else:
                required_samples_per_feature = estimate_feature_samples(res["var"] * res["num_samples"],
                                                                        alpha=(1 - args.confidence_interval),
                                                                        max_abs_error=args.max_abs_error)
                required_samples_per_feature -= res["num_samples"]
                taken_or_estimated_samples = int(
                    res["taken_samples"] + torch.sum(required_samples_per_feature[required_samples_per_feature > 0])
                )

            logging.info(f"[{args.method}] {'taken' if compute_accurately else '(estimated) required'} "
                         f"samples: {taken_or_estimated_samples}")

        logging.info(f"[{args.method}] Time taken: {t2 - t1:.2f}s")
        sequence_tokens = res["input"]

        gen_samples = []
        if args.return_generated_samples:
            for curr_samples in res["samples"]:
                if curr_samples is None:  # non-perturbable feature
                    gen_samples.append([])
                else:
                    gen_samples.append(model.from_internal(curr_samples, skip_special_tokens=False,
                                                           take_as_single_sequence=True))

        method_data.add_example(sequence=sequence_tokens, label=predicted_label, probas=probas[0].tolist(),
                                actual_label=actual_label, custom_features=curr_features,
                                importances=res["importance"].tolist(),
                                num_estimated_samples=res["taken_samples"], time_taken=(t2 - t1),
                                samples=gen_samples,
                                model_scores=[[] if scores is None else scores
                                              for scores in res["scores"]] if args.return_model_scores else [],
                                **side_results)

        if (1 + idx_example) % args.save_every_n_examples == 0:
            logging.info(f"Saving data to {args.experiment_dir}")
            method_data.save(args.experiment_dir, file_name=f"{args.method}_data.json")

            highlight_plot(method_data.sequences, method_data.importances,
                           pred_labels=[method_data.possible_labels[i] for i in method_data.pred_labels],
                           actual_labels=[method_data.possible_labels[i] for i in method_data.actual_labels],
                           custom_features=method_data.custom_features,
                           path=os.path.join(args.experiment_dir, f"{args.method}_importances.html"))
