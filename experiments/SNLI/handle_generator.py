from experiments.SNLI.data import IDX_TO_LABEL
from explain_nlp.methods.generation import BertForMaskedLMGenerator, GPTLMGenerator, GPTControlledLMGenerator


def load_generator(args):
    # IME does not require a generator and loading it would be a waste of a lot of memory
    if args.method in ["ime", "sequential_ime"]:
        return None, {}

    masked_at_once = args.masked_at_once if args.masked_at_once is not None else 1

    generator_description = {
        "type": args.generator_type,
        "max_seq_len": args.generator_max_seq_len,
        "top_p": args.top_p,
        "handle": args.generator_dir,
        "masked_at_once": args.masked_at_once
    }
    if args.generator_type == "bert_mlm":
        generator = BertForMaskedLMGenerator(tokenizer_name=args.generator_dir,
                                             model_name=args.generator_dir,
                                             batch_size=args.generator_batch_size,
                                             max_seq_len=args.generator_max_seq_len,
                                             device="cpu" if args.use_cpu else "cuda",
                                             top_p=args.top_p,
                                             masked_at_once=masked_at_once,
                                             p_ensure_different=args.p_ensure_different,
                                             is_controlled_lm=args.controlled)
    elif args.generator_type == "gpt_lm":
        generator = GPTLMGenerator(tokenizer_name=args.generator_dir,
                                   model_name=args.generator_dir,
                                   batch_size=args.generator_batch_size,
                                   max_seq_len=args.generator_max_seq_len,
                                   device="cpu" if args.use_cpu else "cuda",
                                   top_p=args.top_p,
                                   masked_at_once=masked_at_once,
                                   p_ensure_different=args.p_ensure_different)
    elif args.generator_type == "gpt_controlled_lm":
        generator = GPTControlledLMGenerator(tokenizer_name=args.generator_dir,
                                             model_name=args.generator_dir,
                                             possible_labels=[IDX_TO_LABEL[i] for i in sorted(IDX_TO_LABEL.keys())],
                                             batch_size=args.generator_batch_size,
                                             max_seq_len=args.generator_max_seq_len,
                                             device="cpu" if args.use_cpu else "cuda",
                                             top_p=args.top_p,
                                             masked_at_once=masked_at_once,
                                             p_ensure_different=args.p_ensure_different)
    else:
        raise NotImplementedError(f"'{args.generator_type}' is not a supported generator type")

    return generator, generator_description
