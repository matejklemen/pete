from experiments.IMDb_sentiment.data import IDX_TO_LABEL
from explain_nlp.methods.generation import BertForMaskedLMGenerator, GPTLMGenerator, GPTControlledLMGenerator


def load_generator(args):
    # IME does not require a generator and loading it would be a waste of a lot of memory
    if args.method in ["ime", "sequential_ime", "whole_word_ime"]:
        return None, {}

    generator_description = {
        "type": args.generator_type,
        "max_seq_len": args.generator_max_seq_len,
        "top_p": args.top_p,
        "handle": args.generator_dir,
    }
    if args.generator_type == "bert_mlm":
        generator = BertForMaskedLMGenerator(tokenizer_name=args.generator_dir,
                                             model_name=args.generator_dir,
                                             batch_size=args.generator_batch_size,
                                             max_seq_len=args.generator_max_seq_len,
                                             device="cpu" if args.use_cpu else "cuda",
                                             strategy=args.strategy,
                                             top_p=args.top_p,
                                             top_k=args.top_k,
                                             threshold=args.threshold,
                                             is_controlled_lm=args.controlled)
    elif args.generator_type == "gpt_lm":
        generator = GPTLMGenerator(tokenizer_name=args.generator_dir,
                                   model_name=args.generator_dir,
                                   batch_size=args.generator_batch_size,
                                   max_seq_len=args.generator_max_seq_len,
                                   device="cpu" if args.use_cpu else "cuda",
                                   top_p=args.top_p)
    elif args.generator_type == "gpt_controlled_lm":
        generator = GPTControlledLMGenerator(tokenizer_name=args.generator_dir,
                                             model_name=args.generator_dir,
                                             possible_labels=[IDX_TO_LABEL[i] for i in sorted(IDX_TO_LABEL.keys())],
                                             batch_size=args.generator_batch_size,
                                             max_seq_len=args.generator_max_seq_len,
                                             device="cpu" if args.use_cpu else "cuda",
                                             top_p=args.top_p)
    else:
        raise NotImplementedError(f"'{args.generator_type}' is not a supported generator type")

    return generator, generator_description
