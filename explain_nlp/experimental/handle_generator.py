from typing import List, Optional

from explain_nlp.generation.generation_lstm import ContextualBiLSTMLMGenerator, LSTMConditionallyIndependentGenerator
from explain_nlp.generation.generation_transformers import BertForMaskedLMGenerator, GPTLMGenerator, \
    GPTControlledLMGenerator, BertForControlledMaskedLMGenerator, SimplifiedBertForMaskedLMGenerator, \
    RobertaForMaskedLMGenerator


def load_generator(args, clm_labels: Optional[List[str]] = None, **kwargs):
    # base IME and LIME do not require a generator and loading it could be a waste of a lot of memory
    if args.method in ["ime", "lime"]:
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
                                             threshold=args.threshold)
    elif args.generator_type == "roberta_mlm":
        generator = RobertaForMaskedLMGenerator(tokenizer_name=args.generator_dir,
                                                model_name=args.generator_dir,
                                                batch_size=args.generator_batch_size,
                                                max_seq_len=args.generator_max_seq_len,
                                                device="cpu" if args.use_cpu else "cuda",
                                                strategy=args.strategy,
                                                top_p=args.top_p,
                                                top_k=args.top_k,
                                                threshold=args.threshold)
    elif args.generator_type == "bert_simplified_mlm":
        generator = SimplifiedBertForMaskedLMGenerator(tokenizer_name=args.generator_dir,
                                                       model_name=args.generator_dir,
                                                       batch_size=args.generator_batch_size,
                                                       max_seq_len=args.generator_max_seq_len,
                                                       device="cpu" if args.use_cpu else "cuda",
                                                       strategy=args.strategy,
                                                       top_p=args.top_p,
                                                       top_k=args.top_k,
                                                       threshold=args.threshold)
    elif args.generator_type == "bert_cmlm":
        generator = BertForControlledMaskedLMGenerator(tokenizer_name=args.generator_dir,
                                                       model_name=args.generator_dir,
                                                       control_labels=clm_labels,
                                                       label_weights=kwargs.get("label_weights", None),
                                                       batch_size=args.generator_batch_size,
                                                       max_seq_len=args.generator_max_seq_len,
                                                       device="cpu" if args.use_cpu else "cuda",
                                                       strategy=args.strategy,
                                                       top_p=args.top_p,
                                                       top_k=args.top_k,
                                                       threshold=args.threshold,
                                                       unique_dropout=args.unique_dropout)
    elif args.generator_type == "gpt_lm":
        generator = GPTLMGenerator(tokenizer_name=args.generator_dir,
                                   model_name=args.generator_dir,
                                   batch_size=args.generator_batch_size,
                                   max_seq_len=args.generator_max_seq_len,
                                   device="cpu" if args.use_cpu else "cuda",
                                   strategy=args.strategy,
                                   top_p=args.top_p,
                                   top_k=args.top_k,
                                   threshold=args.threshold)
    elif args.generator_type == "gpt_clm":
        generator = GPTControlledLMGenerator(tokenizer_name=args.generator_dir,
                                             model_name=args.generator_dir,
                                             control_labels=clm_labels,
                                             label_weights=kwargs.get("label_weights", None),
                                             batch_size=args.generator_batch_size,
                                             max_seq_len=args.generator_max_seq_len,
                                             device="cpu" if args.use_cpu else "cuda",
                                             strategy=args.strategy,
                                             top_p=args.top_p,
                                             top_k=args.top_k,
                                             threshold=args.threshold)
    elif args.generator_type == "cblstm_lm":
        generator = ContextualBiLSTMLMGenerator(tokenizer_name=args.generator_dir,
                                                model_name=args.generator_dir,
                                                max_seq_len=args.generator_max_seq_len,
                                                batch_size=args.generator_batch_size,
                                                device="cpu" if args.use_cpu else "cuda",
                                                strategy=args.strategy,
                                                top_p=args.top_p,
                                                top_k=args.top_k,
                                                threshold=args.threshold)
    elif args.generator_type == "aelstm_lm":
        generator = LSTMConditionallyIndependentGenerator(tokenizer_name=args.generator_dir,
                                                          model_name=args.generator_dir,
                                                          max_seq_len=args.generator_max_seq_len,
                                                          batch_size=args.generator_batch_size,
                                                          device="cpu" if args.use_cpu else "cuda",
                                                          strategy=args.strategy,
                                                          top_p=args.top_p,
                                                          top_k=args.top_k,
                                                          threshold=args.threshold)
    else:
        raise NotImplementedError(f"'{args.generator_type}' is not a supported generator type")

    return generator, generator_description
