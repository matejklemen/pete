import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_type", type=str, choices=["accurate_importances", "required_samples"],
                    default="required_samples")
parser.add_argument("--method", type=str, default="ime",
                    choices=["ime", "sequential_ime", "whole_word_ime", "ime_mlm", "ime_dependent_mlm"])
parser.add_argument("--custom_features", type=str, default=None,
                    choices=[None, "words", "sentences", "depparse_simple", "depparse_depth"])
parser.add_argument("--min_samples_per_feature", type=int, default=10,
                    help="Minimum number of samples that get created for each feature for initial variance estimation")
parser.add_argument("--confidence_interval", type=float, default=0.99)
parser.add_argument("--max_abs_error", type=float, default=0.01)
parser.add_argument("--return_generated_samples", action="store_true")
parser.add_argument("--return_model_scores", action="store_true")

parser.add_argument("--train_path", type=str, default="/home/matej/Documents/data/snli/snli_1.0_train.txt")
parser.add_argument("--test_path", type=str, default="/home/matej/Documents/data/snli/snli_1.0_test_xs.txt")

parser.add_argument("--model_dir", type=str, default="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased")
parser.add_argument("--model_max_seq_len", type=int, default=41)
parser.add_argument("--model_max_words", type=int, default=39)
parser.add_argument("--model_batch_size", type=int, default=2)

parser.add_argument("--generator_type", type=str, default="bert_cmlm")
parser.add_argument("--controlled", action="store_true",
                    help="Whether to use controlled LM/MLM for generation")
parser.add_argument("--generator_dir", type=str, default="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert_snli_clm_best",
                    help="Path or handle of model to be used as a language modeling generator")
parser.add_argument("--generator_batch_size", type=int, default=2)
parser.add_argument("--generator_max_seq_len", type=int, default=41)
parser.add_argument("--num_generated_samples", type=int, default=10)
parser.add_argument("--top_p", type=float, default=None)

# Experimental (only in Bert MLM generator) for now
parser.add_argument("--strategy", type=str, choices=["top_k", "top_p", "threshold", "greedy"], default="greedy")
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--threshold", type=float, default=0.1)
parser.add_argument("--unique_dropout", type=float, default=0.0)

parser.add_argument("--seed_start_with_ground_truth", action="store_true")
parser.add_argument("--reset_seed_after_first", action="store_true")

parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--save_every_n_examples", type=int, default=1,
                    help="Save experiment data every N examples in order to avoid losing data on longer computations")

parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of GPU")
parser.add_argument("--verbose", action="store_true")

parser.add_argument("--start_from", type=int, default=None, help="From which example onwards to do computation")
parser.add_argument("--until", type=int, default=None, help="Until which example to do computation")
