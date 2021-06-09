import logging
import os
import sys
from time import time

import stanza
import torch

from explain_nlp.experimental.arguments import methods_parser, runtime_parse_args, log_arguments
from explain_nlp.experimental.core import MethodData
from explain_nlp.experimental.data import load_sst2, LABEL_TO_IDX, TransformerSeqDataset, IDX_TO_LABEL
from explain_nlp.experimental.handle_explainer import load_explainer
from explain_nlp.experimental.handle_generator import load_generator
from explain_nlp.methods.custom_units import WordExplainer, SentenceExplainer, DependencyTreeExplainer
from explain_nlp.methods.hybrid import create_uniform_weights
from explain_nlp.methods.utils import estimate_feature_samples
from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification
from explain_nlp.visualizations.highlight import highlight_plot

STANZA_BATCH_SIZE = 1024

if __name__ == "__main__":
    args = methods_parser.parse_args()
    args = runtime_parse_args(args)
    compute_accurately = args.method_class == "ime" and args.experiment_type == "accurate_importances"

    # Set up logging to file and stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout),
                         logging.FileHandler(os.path.join(args.experiment_dir, "experiment.log"))]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)

    log_arguments(args)

    nlp = stanza.Pipeline(lang="en", processors="tokenize", use_gpu=not args.use_cpu, tokenize_no_ssplit=True)
    pretokenized_test_data = None

    model = InterpretableBertForSequenceClassification(tokenizer_name=args.model_dir,
                                                       model_name=args.model_dir,
                                                       batch_size=args.model_batch_size,
                                                       max_seq_len=args.model_max_seq_len,
                                                       device="cpu" if args.use_cpu else "cuda")
    model_description = {"type": "bert", "max_seq_len": args.model_max_seq_len, "handle": args.model_dir}
    generator, gen_description = load_generator(args)

    df_test = load_sst2(args.test_path)
    if args.custom_features is not None:
        logging.info("Tokenizing test examples with Stanza")
        pretokenized_test_data = []
        for idx_subset in range((df_test.shape[0] + STANZA_BATCH_SIZE - 1) // STANZA_BATCH_SIZE):
            s, e = idx_subset * STANZA_BATCH_SIZE, (1 + idx_subset) * STANZA_BATCH_SIZE
            for sent in nlp("\n\n".join(df_test["sentence"].iloc[s: e].values)).sentences:
                pretokenized_test_data.append([token.words[0].text for token in sent.tokens])

    used_data = {"test_path": args.test_path}
    used_sample_data = None
    data_weights = None
    if args.method in {"ime", "ime_hybrid"}:
        logging.info("Loading training dataset as sampling data")
        used_data["train_path"] = args.train_path
        df_train = load_sst2(args.train_path).sample(frac=1.0).reset_index(drop=True)
        used_tokenizer = model.tokenizer if args.method == "ime" else generator.tokenizer
        train_set = TransformerSeqDataset.build(df_train["sentence"].tolist(),
                                                labels=df_train["label"].apply(
                                                    lambda label_str: LABEL_TO_IDX["sst-2"][label_str]).tolist(),
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
                             possible_labels=[IDX_TO_LABEL["sst-2"][i] for i in sorted(IDX_TO_LABEL["sst-2"])],
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
        if args.custom_features.startswith("dependency_parsing"):
            nlp = stanza.Pipeline(lang="en", processors="tokenize,lemma,pos,depparse")
        else:
            nlp = stanza.Pipeline(lang="en", processors="tokenize")

        if args.custom_features == "words":
            method = WordExplainer(method, stanza_pipeline=nlp)
        elif args.custom_features == "sentences":
            method = SentenceExplainer(method, stanza_pipeline=nlp)
        elif args.custom_features == "dependency_parsing":
            method = DependencyTreeExplainer(method, stanza_pipeline=nlp)

    logging.info(f"Running computation from example#{start_from} (inclusive) to example#{until} (exclusive)")
    for idx_example, input_seq in enumerate(df_test.iloc[start_from: until]["sentence"].tolist(),
                                            start=start_from):
        curr_example = df_test.iloc[idx_example]

        encoded_example = model.to_internal(
            text_data=[pretokenized_test_data[idx_example] if args.custom_features is not None else input_seq],
            is_split_into_units=(args.custom_features is not None)
        )

        probas = model.score(input_ids=encoded_example["input_ids"],
                             **{k: encoded_example["aux_data"][k] for k in ["token_type_ids", "attention_mask"]})
        predicted_label = int(torch.argmax(probas))
        actual_label = int(LABEL_TO_IDX["sst-2"][curr_example["label"]])

        side_results = {}
        t1 = time()
        if args.method_class == "lime":
            res = method.explain_text(input_seq, label=predicted_label,
                                      num_samples=num_taken_samples,
                                      explanation_length=args.explanation_length)
        else:
            # Run method in order to obtain an estimate of the feature variance
            res = method.explain_text(input_seq, label=predicted_label,
                                      min_samples_per_feature=num_taken_samples)

            # Estimate the number of samples that need to be taken to satisfy constraints:
            # in case less samples are required than were already taken, we take `max(2, <num. required samples>)`
            # as the required number of samples for that feature
            required_samples_per_feature = estimate_feature_samples(res["var"] * res["num_samples"],
                                                                    alpha=(1 - args.confidence_interval),
                                                                    max_abs_error=args.max_abs_error).long()
            need_additional = required_samples_per_feature > res["num_samples"]
            need_less = torch.logical_and(torch.eq(res["num_samples"], num_taken_samples),
                                          torch.le(required_samples_per_feature, res["num_samples"]))
            res["num_samples"][need_additional] = required_samples_per_feature[need_additional]
            res["num_samples"][need_less] = torch.max(required_samples_per_feature[need_less], torch.tensor(2))
            taken_or_estimated_samples = int(torch.sum(res["num_samples"]))

            # Run method second time in order to obtain accurate importances, making use of batched computation
            if compute_accurately:
                res = method.explain_text(input_seq, label=predicted_label,
                                          exact_samples_per_feature=res["num_samples"].unsqueeze(0))

            side_results["variances"] = res["var"].tolist()
            side_results["num_samples"] = res["num_samples"].tolist()
            logging.info(f"[{args.method}] {'Taken' if compute_accurately else '(Estimated) required'} "
                         f"samples: {taken_or_estimated_samples}")

        t2 = time()
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
                                actual_label=actual_label,
                                custom_features=res.get("custom_features", None),
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







