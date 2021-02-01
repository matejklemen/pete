import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_type", type=str, default="required_samples",
                    choices=["accurate_importances", "required_samples"])
parser.add_argument("--method", type=str, default="ime_hybrid",
                    choices=["ime", "sequential_ime", "whole_word_ime", "ime_mlm", "ime_dependent_mlm", "ime_hybrid"])
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

parser.add_argument("--generator_type", type=str, default="bert_mlm",
                    choices=["bert_mlm", "bert_cmlm", "gpt_lm", "gpt_clm"])
parser.add_argument("--generator_dir", type=str, default="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert-base-uncased-snli-mlm",
                    help="Path or handle of model to be used as a language modeling generator")
parser.add_argument("--generator_batch_size", type=int, default=8)
parser.add_argument("--generator_max_seq_len", type=int, default=41)
parser.add_argument("--num_generated_samples", type=int, default=100)
parser.add_argument("--generate_cover", action="store_true", default=False,
                    help="Take all relevant tokens instead of sampling one token from renormalized distributions")
parser.add_argument("--is_aligned_vocabulary", action="store_true")

parser.add_argument("--strategy", type=str, default="top_p",
                    choices=["top_k", "top_p", "threshold", "greedy"])
parser.add_argument("--top_p", type=float, default=0.999)
parser.add_argument("--top_k", type=int, default=3)
parser.add_argument("--threshold", type=float, default=0.1)
parser.add_argument("--unique_dropout", type=float, default=0.0)

parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of GPU")
parser.add_argument("--verbose", action="store_true")

parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--save_every_n_examples", type=int, default=1,
                    help="Save experiment data every N examples in order to avoid losing data on longer computations")
parser.add_argument("--start_from", type=int, default=None, help="From which example onwards to do computation")
parser.add_argument("--until", type=int, default=None, help="Until which example to do computation")
