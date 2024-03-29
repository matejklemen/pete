import argparse
import logging
import os
import sys
from time import time

import stanza
import torch

from explain_nlp.experimental.arguments import runtime_parse_args, log_arguments
from explain_nlp.experimental.core import MethodData
from explain_nlp.experimental.data import load_dataset, TransformerSeqPairDataset, IDX_TO_LABEL, LABEL_TO_IDX
from explain_nlp.experimental.handle_explainer import load_explainer
from explain_nlp.experimental.handle_generator import load_generator
from explain_nlp.experimental.handle_model import load_model
from explain_nlp.methods.custom_units import WordExplainer, SentenceExplainer, DependencyTreeExplainer
from explain_nlp.methods.ime_lm import create_uniform_weights
from explain_nlp.methods.utils import estimate_feature_samples
from explain_nlp.visualizations.highlight import highlight_plot

STANZA_BATCH_SIZE = 1024

general_parser = argparse.ArgumentParser(add_help=False)
general_parser.add_argument("--experiment_dir", type=str, default=None)
general_parser.add_argument("--save_every_n_examples", type=int, default=1,
                            help="Save experiment data every N examples in order to avoid losing data on longer computations")
general_parser.add_argument("--start_from", type=int, default=None, help="From which example onwards to do computation")
general_parser.add_argument("--until", type=int, default=None, help="Until which example to do computation")
general_parser.add_argument("--random_seed", type=int, default=None)
general_parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of GPU")

general_parser.add_argument("--custom_features", type=str, default=None,
                            choices=[None, "words", "sentences", "dependency_parsing"])
general_parser.add_argument("--return_generated_samples", action="store_true")
general_parser.add_argument("--return_model_scores", action="store_true")
general_parser.add_argument("--test_path", type=str,
                            default="/home/matej/Documents/data/XNLI-1.0/subsample/xnli_test5_xs.tsv")

general_parser.add_argument("--model_type", type=str, default="xlmr_sequence",
                            choices=["bert_sequence", "xlmr_sequence"])
general_parser.add_argument("--model_dir", type=str,
                            default="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/classifiers/xnli-cross-all-languages-xlm-roberta-base-2e-5-maxlen109")
general_parser.add_argument("--model_max_seq_len", type=int, default=73)
general_parser.add_argument("--model_batch_size", type=int, default=4)

general_parser.add_argument("--generator_type", type=str, default="xlmr_mlm",
                            choices=["bert_mlm", "bert_cmlm", "xlmr_mlm", "xlmr_cmlm",
                                     "gpt_lm", "gpt_clm",
                                     "cblstm_lm"])
general_parser.add_argument("--generator_dir", type=str,
                            help="Path or handle of model to be used as a language modeling generator",
                            default="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/language_models/xlm-roberta-xnli-mlm")
general_parser.add_argument("--generator_batch_size", type=int, default=4)
general_parser.add_argument("--generator_max_seq_len", type=int, default=73)
general_parser.add_argument("--strategy", type=str, default="top_p",
                            choices=["top_k", "top_p", "greedy"])
general_parser.add_argument("--top_p", type=float, default=0.95)
general_parser.add_argument("--top_k", type=int, default=3)
general_parser.add_argument("--use_mcd", action="store_true",
                            help="If set, leave dropout on during generation. This should introduce additional "
                                 "variance into the generated sequences")
general_parser.add_argument("--shared_vocabulary", action="store_true",
                            help="If set, methods assume the model and generator use same vocabulary and do not need "
                                 "conversion between representations")
general_parser.add_argument("--mask_in_advance", action="store_true")
general_parser.add_argument("--use_contrastive_control", action="store_true",
                            help="If set, set all control labels to a semantically most different label, e.g., "
                                 "if explanaining a hateful example, all control labels would be set to [CLEAN]")

methods_parser = argparse.ArgumentParser()
subparsers = methods_parser.add_subparsers(dest="method_class")

""" Specific arguments for IME """
ime_parser = subparsers.add_parser("ime", parents=[general_parser])
ime_parser.add_argument("--experiment_type", type=str, default="required_samples",
                        choices=["accurate_importances", "required_samples", "max_samples", "min_samples_per_feature"])
ime_parser.add_argument("--method", type=str, default="ime",
                        choices=["ime", "ime_elm", "ime_ilm", "ime_hybrid"])
ime_parser.add_argument("--min_samples_per_feature", type=int, default=10,
                        help="Minimum number of samples that get created for each feature for initial variance estimation")
ime_parser.add_argument("--max_samples", type=int, default=None)
ime_parser.add_argument("--confidence_interval", type=float, default=0.95)
ime_parser.add_argument("--max_abs_error", type=float, default=0.01)
ime_parser.add_argument("--train_path", type=str, default="/home/matej/Documents/data/XNLI-1.0/xnli.dev.tsv")
ime_parser.add_argument("--num_generated_samples", type=int, default=100,
                        help="Number of samples to generate with generator, when using IME with external LM")

""" Specific arguments for LIME """
lime_parser = subparsers.add_parser("lime", parents=[general_parser])
lime_parser.add_argument("--method", type=str, default="lime",
                         choices=["lime", "lime_lm"])
lime_parser.add_argument("--num_samples", type=int, default=100,
                         help="Number of samples to take when generating neighbourhood")
lime_parser.add_argument("--explanation_length", type=int, default=None)
lime_parser.add_argument("--kernel_width", type=float, default=1.0)


def load_custom_explainer(explanation_method, custom_features, lang):
    """
    Args:
        explanation_method:
            Base explanation method (e.g., IME, LIME+LM, ...) to be wrapped with custom features
        custom_features:
            Type of custom features
        lang:
            Language of the Stanza pipeline to be loaded
    """
    if custom_features == "dependency_parsing":
        nlp = stanza.Pipeline(lang=lang, processors="tokenize,lemma,pos,depparse")
    else:
        nlp = stanza.Pipeline(lang=lang, processors="tokenize")

    if custom_features == "words":
        wrapped_method = WordExplainer(explanation_method, stanza_pipeline=nlp)
    elif custom_features == "sentences":
        wrapped_method = SentenceExplainer(explanation_method, stanza_pipeline=nlp)
    elif custom_features == "dependency_parsing":
        wrapped_method = DependencyTreeExplainer(explanation_method, stanza_pipeline=nlp)
    else:
        raise NotImplementedError(custom_features)

    return wrapped_method


if __name__ == "__main__":
    args = methods_parser.parse_args()

    dataset_name = "xnli"  # some arguments can be automatically extracted from presets
    args = runtime_parse_args(args)
    compute_required = args.method_class == "ime" and args.experiment_type == "required_samples"
    eff_max_samples = args.max_samples \
        if (args.method_class == "ime" and args.experiment_type == "max_samples") else None

    # Set up logging to file and stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout),
                         logging.FileHandler(os.path.join(args.experiment_dir, "experiment.log"))]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)

    log_arguments(args)

    df_test = load_dataset(dataset_name, args.test_path)

    pretokenized_test_data = None
    if args.custom_features is not None:
        # Initialize to dummy values, override later (needed because examples might not be sorted by language)
        pretokenized_test_data = [None for _ in range(df_test.shape[0])]
        for lang, group in df_test.groupby("language"):
            logging.info(f"Tokenizing '{lang}' test examples with Stanza")
            nlp = stanza.Pipeline(lang=lang, processors="tokenize", use_gpu=(not args.use_cpu), tokenize_no_ssplit=True)

            indices = group.index.tolist()
            for idx_subset in range((group.shape[0] + STANZA_BATCH_SIZE - 1) // STANZA_BATCH_SIZE):
                s, e = idx_subset * STANZA_BATCH_SIZE, (1 + idx_subset) * STANZA_BATCH_SIZE

                for position, s0, s1 in zip(indices[s: e],
                                            nlp("\n\n".join(group["sentence1"].iloc[s: e].values)).sentences,
                                            nlp("\n\n".join(group["sentence2"].iloc[s: e].values)).sentences):
                    pretokenized_test_data[position] = (
                        [token.text for token in s0.words],
                        [token.text for token in s1.words]
                    )

    raw_examples = list(map(lambda row: tuple(row), df_test[["sentence1", "sentence2"]].values))

    model = load_model(model_type=args.model_type, model_name=args.model_dir, tokenizer_name=args.model_dir,
                       batch_size=args.model_batch_size, max_seq_len=args.model_max_seq_len,
                       device="cpu" if args.use_cpu else "cuda")
    model_description = {"type": args.model_type, "handle": args.model_dir, "max_seq_len": args.model_max_seq_len}

    generator = load_generator(args, clm_labels=["<ENTAILMENT>", "<NEUTRAL>", "<CONTRADICTION>"])
    if generator is not None:
        generator_description = {
            "type": args.model_type,
            "handle": args.model_dir,
            "max_seq_len": args.model_max_seq_len
        }
    else:
        generator_description = {}

    used_data = {"test_path": args.test_path}
    used_sample_data, data_weights = None, None
    xnli_lang_to_indices = {}
    if args.method in {"ime", "ime_hybrid"}:
        logging.info("Loading training dataset as sampling data")
        used_data["train_path"] = args.train_path
        df_train = load_dataset(dataset_name, args.train_path).sample(frac=1.0).reset_index(drop=True)
        for lang, group in df_train.groupby("language"):
            xnli_lang_to_indices[lang] = group.index.tolist()

        # Hybrid IME estimates feature importance in generator's representation
        used_tokenizer = model.tokenizer if args.method == "ime" else generator.tokenizer
        used_max_seq_len = model.max_seq_len if args.method == "ime" else generator.max_seq_len
        train_set = TransformerSeqPairDataset.build_dataset(dataset_name, df_train,
                                                            tokenizer=used_tokenizer,
                                                            max_seq_len=used_max_seq_len)

        data_weights = create_uniform_weights(train_set.input_ids, train_set.special_tokens_mask)
        used_sample_data = train_set.input_ids

    contrastive_labels = None
    if args.use_contrastive_control:
        logging.info("Using contrastive control labels!")
        contrastive_labels = {
            LABEL_TO_IDX["xnli"]["entailment"]: LABEL_TO_IDX["xnli"]["contradiction"],
            LABEL_TO_IDX["xnli"]["neutral"]: LABEL_TO_IDX["xnli"]["contradiction"],
            LABEL_TO_IDX["xnli"]["contradiction"]: LABEL_TO_IDX["xnli"]["entailment"],
        }

    base_method, method_type = load_explainer(model=model, generator=generator,
                                              used_sample_data=used_sample_data, data_weights=data_weights,
                                              contrastive_labels=contrastive_labels, **vars(args))
    method = base_method

    # Container that wraps debugging data and a lot of repetitive appends
    method_data = MethodData(method_type=method_type,
                             model_description=model_description, generator_description=generator_description,
                             possible_labels=[IDX_TO_LABEL[dataset_name][i] for i in sorted(IDX_TO_LABEL[dataset_name])],
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

    # language of currently explained instance - used to determine the right Stanza pipeline to use
    curr_lang = None

    logging.info(f"Running computation from example#{start_from} (inclusive) to example#{until} (exclusive)")
    for idx_example in range(start_from, until):
        lang_of_example = df_test.iloc[idx_example]["language"]
        if args.custom_features is not None and lang_of_example != curr_lang:
            logging.info(f"Reloading explainer for language '{lang_of_example}'")
            method = load_custom_explainer(explanation_method=base_method,
                                           custom_features=args.custom_features,
                                           lang=lang_of_example)

        encoded_example = model.to_internal(
            text_data=[pretokenized_test_data[idx_example]
                       if args.custom_features is not None else raw_examples[idx_example]],
            is_split_into_units=args.custom_features is not None
        )

        probas = model.score(input_ids=encoded_example["input_ids"],
                             **{k: encoded_example["aux_data"][k] for k in ["attention_mask"]})
        predicted_label = int(torch.argmax(probas))
        actual_label = int(LABEL_TO_IDX[dataset_name][df_test.iloc[idx_example]["gold_label"]])

        side_results = {}
        t1 = time()
        if args.method_class == "lime":
            res = method.explain_text(raw_examples[idx_example], label=predicted_label,
                                      num_samples=args.num_samples,
                                      explanation_length=args.explanation_length)
        else:  # IME*
            # Only use sampling data corresponding to the language of the explained instance
            if lang_of_example != curr_lang and args.method in {"ime", "ime_hybrid"}:
                logging.info(f"Updating sampling data to samples for '{lang_of_example}'")
                curr_indices = xnli_lang_to_indices[lang_of_example]
                method.update_sample_data(used_sample_data[curr_indices], data_weights[curr_indices])

            res = method.explain_text(raw_examples[idx_example], label=predicted_label,
                                      min_samples_per_feature=args.min_samples_per_feature,
                                      max_samples=eff_max_samples)

            if compute_required:
                # Estimate the number of samples that need to be taken to satisfy constraints:
                # in case less samples are required than were already taken, we take min_samples_per_feature
                required_samples_per_feature = estimate_feature_samples(res["var"] * res["num_samples"],
                                                                        alpha=(1 - args.confidence_interval),
                                                                        max_abs_error=args.max_abs_error).long()
                need_additional = required_samples_per_feature > res["num_samples"]
                res["num_samples"][need_additional] = required_samples_per_feature[need_additional]

            taken_or_estimated_samples = int(torch.sum(res["num_samples"]))

            side_results["variances"] = res["var"].tolist()
            side_results["num_samples"] = res["num_samples"].tolist()
            logging.info(f"[{args.method}] {'(Estimated) required' if compute_required else 'Taken'} "
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

        curr_lang = lang_of_example

        if (1 + idx_example) % args.save_every_n_examples == 0:
            logging.info(f"Saving data to {args.experiment_dir}")
            method_data.save(args.experiment_dir, file_name=f"{args.method}_data.json")

            highlight_plot(method_data.sequences, method_data.importances,
                           pred_labels=[method_data.possible_labels[i] for i in method_data.pred_labels],
                           actual_labels=[method_data.possible_labels[i] for i in method_data.actual_labels],
                           custom_features=method_data.custom_features,
                           path=os.path.join(args.experiment_dir, f"{args.method}_importances.html"))
