import argparse
import json
import logging
import os
import time
import warnings

""" Parser that has common arguments for methods. `add_help=False` because otherwise flags overlap """
general_parser = argparse.ArgumentParser(add_help=False)
general_parser.add_argument("--experiment_dir", type=str, default=None)
general_parser.add_argument("--save_every_n_examples", type=int, default=1,
                            help="Save experiment data every N examples in order to avoid losing data on longer computations")
general_parser.add_argument("--start_from", type=int, default=None, help="From which example onwards to do computation")
general_parser.add_argument("--until", type=int, default=None, help="Until which example to do computation")
general_parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of GPU")

general_parser.add_argument("--custom_features", type=str, default=None,
                            choices=[None, "words", "sentences", "depparse_simple", "depparse_depth"])
general_parser.add_argument("--return_generated_samples", action="store_true")
general_parser.add_argument("--return_model_scores", action="store_true")
general_parser.add_argument("--test_path", type=str,
                            default="/home/matej/Documents/data/snli/snli_1.0_test_xs.txt")

general_parser.add_argument("--model_dir", type=str,
                            default="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased")
general_parser.add_argument("--model_max_seq_len", type=int, default=41)
general_parser.add_argument("--model_max_words", type=int, default=39)
general_parser.add_argument("--model_batch_size", type=int, default=2)

general_parser.add_argument("--generator_type", type=str, default="bert_simplified_mlm",
                            choices=["bert_mlm", "roberta_mlm",
                                     "bert_simplified_mlm", "bert_cmlm", "bert_simplified_cmlm",
                                     "gpt_lm", "gpt_clm",
                                     "cblstm_lm"])
general_parser.add_argument("--generator_dir", type=str,
                            help="Path or handle of model to be used as a language modeling generator",
                            default="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert-base-uncased-snli-mlm")
general_parser.add_argument("--generator_batch_size", type=int, default=8)
general_parser.add_argument("--generator_max_seq_len", type=int, default=41)
general_parser.add_argument("--strategy", type=str, default="top_p",
                            choices=["top_k", "top_p", "threshold", "greedy"])
general_parser.add_argument("--top_p", type=float, default=0.95)
general_parser.add_argument("--top_k", type=int, default=3)
general_parser.add_argument("--threshold", type=float, default=0.1)
general_parser.add_argument("--unique_dropout", type=float, default=0.0)
general_parser.add_argument("--is_aligned_vocabulary", action="store_true")

# TODO: this is only required for QUACKIE experiments, does not need to be in general parser
general_parser.add_argument("--aggregation_strategy", choices=["subword_sum", "subword_max", "sentence"],
                            help="Specifies how to obtain sentence scores from words in QUACKIE experiments: "
                                 "by summing word importance, taking the max word importance or by using sentences as "
                                 "primary explanation units",
                            default="sentence")
# TODO: this is only required for simplified models
general_parser.add_argument("--num_references", type=int, default=10)

methods_parser = argparse.ArgumentParser()
subparsers = methods_parser.add_subparsers(dest="method_class")

""" Specific arguments for IME """
ime_parser = subparsers.add_parser("ime", parents=[general_parser])
ime_parser.add_argument("--experiment_type", type=str, default="required_samples",
                        choices=["accurate_importances", "required_samples"])
ime_parser.add_argument("--method", type=str, default="ime",
                        choices=["ime", "ime_mlm", "ime_dependent_mlm", "ime_hybrid"])
ime_parser.add_argument("--min_samples_per_feature", type=int, default=10,
                        help="Minimum number of samples that get created for each feature for initial variance estimation")
ime_parser.add_argument("--confidence_interval", type=float, default=0.95)
ime_parser.add_argument("--max_abs_error", type=float, default=0.01)
ime_parser.add_argument("--train_path", type=str, default="/home/matej/Documents/data/snli/snli_1.0_train.txt")
ime_parser.add_argument("--num_generated_samples", type=int, default=100,
                        help="Number of samples to generate with generator, when using IME with external LM")

""" Specific arguments for LIME """
lime_parser = subparsers.add_parser("lime", parents=[general_parser])
lime_parser.add_argument("--method", type=str, default="lime",
                         choices=["lime", "lime_lm"])
lime_parser.add_argument("--num_samples", type=int, default=10,
                         help="Number of samples to take when generating neighbourhood")
lime_parser.add_argument("--explanation_length", type=int, default=None)
lime_parser.add_argument("--kernel_width", type=float, default=25.0)


def runtime_parse_args(args):
    """ Common argument maintenance logic, such as sanity checks, creation of experiment dir, etc.. """
    if args.method not in ["ime", "lime"]:
        assert args.generator_dir is not None
        if not os.path.exists(args.generator_dir):
            warnings.warn(f"--generator_dir does not point to a valid directory: '{args.generator_dir}'")

    assert args.save_every_n_examples > 0

    # Use an automatically generated experiment name if it's not provided
    if args.experiment_dir is None:
        ts = time.time()
        if args.method_class == "lime":
            args.experiment_dir = f"exp_{ts}_{args.method}"
        else:  # IME
            args.experiment_dir = f"exp_{ts}_{args.method}_{args.experiment_type}"
        logging.warning(f"--experiment_dir not provided, using '{args.experiment_dir}'")

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)
        logging.info("Created experiment directory '{args.experiment_dir}'")

    with open(os.path.join(args.experiment_dir, "experiment_config.json"), "w") as f:
        json.dump(vars(args), fp=f, indent=4)

    return args


def log_arguments(args):
    for k, v in vars(args).items():
        v_str = str(v)
        v_str = f"...{v_str[-(50-3):]}" if len(v_str) > 50 else v_str
        logging.info(f"|{k:30s}|{v_str:50s}|")
