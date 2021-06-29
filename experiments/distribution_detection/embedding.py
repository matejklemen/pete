import argparse
import json
import logging
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import stanza
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from explain_nlp.experimental.arguments import log_arguments
from explain_nlp.experimental.data import TransformerSeqDataset, TransformerSeqPairDataset, load_dataset, \
    IDX_TO_LABEL, PRESET_COLNAMES
from explain_nlp.experimental.handle_explainer import load_explainer
from explain_nlp.experimental.handle_generator import load_generator
from explain_nlp.generation.decoding import filter_factory
from explain_nlp.methods.features import extract_groups
from explain_nlp.methods.ime_lm import create_uniform_weights
from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification

""" 
    This is copypasted and trimmed from arguments in arguments.py in order to minimize redundant arguments that are 
    stored with the experiments (it becomes really unclear what is actually being used)
"""
general_parser = argparse.ArgumentParser(add_help=False)
general_parser.add_argument("--experiment_dir", type=str, default="debug_snli")
general_parser.add_argument("--random_seed", type=int, default=None)
general_parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of GPU")
general_parser.add_argument("--custom_features", type=str, default=None,
                            choices=[None, "words", "dependency_parsing"])
general_parser.add_argument("--model_dir", type=str,
                            default="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/classifiers/snli_bert_uncased")
general_parser.add_argument("--model_max_seq_len", type=int, default=41)
general_parser.add_argument("--model_batch_size", type=int, default=8)

general_parser.add_argument("--generator_type", type=str, default="bert_mlm",
                            choices=["bert_mlm", "bert_cmlm"])
general_parser.add_argument("--generator_dir", type=str,
                            default="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert-base-uncased-snli-mlm")
general_parser.add_argument("--generator_batch_size", type=int, default=8)
general_parser.add_argument("--generator_max_seq_len", type=int, default=41)

methods_parser = argparse.ArgumentParser()
subparsers = methods_parser.add_subparsers(dest="method_class")

# IME
ime_parser = subparsers.add_parser("ime", parents=[general_parser])
ime_parser.add_argument("--method", type=str, default="ime",
                        choices=["ime", "ime_elm", "ime_ilm", "ime_hybrid"])
ime_parser.add_argument("--train_path", type=str,
                        default="/home/matej/Documents/data/snli/snli_1.0_train.txt")

# LIME
lime_parser = subparsers.add_parser("lime", parents=[general_parser])
lime_parser.add_argument("--method", type=str, default="lime",
                         choices=["lime", "lime_lm"])

# control
control_parser = subparsers.add_parser("control", parents=[general_parser])
control_parser.add_argument("--method", choices=["control"],
                            default="control", help="Fixed argument that is here just for consistency")

STANZA_BATCH_SIZE = 1024


@torch.no_grad()
def bert_embeddings(model: InterpretableBertForSequenceClassification, input_ids, **modeling_kwargs):
    # BERT: pooler_output -> dropout -> linear -> class
    output = model.model.bert(input_ids=input_ids.to(model.device),
                              **{attr: modeling_kwargs[attr].to(model.device)
                                 for attr in ["token_type_ids", "attention_mask"]})
    return output["pooler_output"]  # [num_examples, hidden_size]


def stanza_tokenize(stanza_pipeline, data, dataset_name):
    # _tokenize() tokenizes a batch of examples with Stanza pipeline
    # _unwrap() extracts the tokens into a list or pair of lists (depending on input)
    if dataset_name in ["snli", "mnli", "xnli", "qqp"]:
        col1, col2 = PRESET_COLNAMES[dataset_name]

        def _tokenize(start_batch, end_batch):
            return (stanza_pipeline("\n\n".join(data[col1].iloc[start_batch: end_batch].values)).sentences,
                    stanza_pipeline("\n\n".join(data[col2].iloc[start_batch: end_batch].values)).sentences)

        def _unwrap(tokenized_pair: Tuple):
            return (
                [token.words[0].text for token in tokenized_pair[0].tokens],
                [token.words[0].text for token in tokenized_pair[1].tokens]
            )
    else:
        col = PRESET_COLNAMES[dataset_name][0]

        def _tokenize(start_batch, end_batch):
            return (stanza_pipeline("\n\n".join(data[col].iloc[start_batch: end_batch].values)).sentences,)

        def _unwrap(tokenized_seq):
            return [token.words[0].text for token in tokenized_seq.tokens]

    pretokenized_test_data = []
    for idx_subset in range((data.shape[0] + STANZA_BATCH_SIZE - 1) // STANZA_BATCH_SIZE):
        s, e = idx_subset * STANZA_BATCH_SIZE, (1 + idx_subset) * STANZA_BATCH_SIZE
        for tokenized_data in zip(*_tokenize(s, e)):
            pretokenized_test_data.append(_unwrap(tokenized_data))

    return pretokenized_test_data


if __name__ == "__main__":
    args = methods_parser.parse_args()

    assert os.path.exists(args.experiment_dir), \
        "--experiment_dir must point to a valid directory. Please run sample.py first in order to create it"

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    with open(os.path.join(args.experiment_dir, "config.json"), "r", encoding="utf-8") as f:
        experiment_config = json.load(f)
        dataset_name = experiment_config["dataset"]

    # Load sample data
    sample_path = os.path.join(args.experiment_dir, "sample.csv")
    assert os.path.exists(sample_path)
    df_sample = pd.read_csv(sample_path)

    mini_experiment_path = os.path.join(args.experiment_dir, args.method)
    if not os.path.exists(mini_experiment_path):
        os.makedirs(mini_experiment_path)

    with open(os.path.join(mini_experiment_path, "embedding_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), fp=f, indent=4)

    # Set up logging to file and stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout),
                         logging.FileHandler(os.path.join(mini_experiment_path, "experiment.log"))]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)
    log_arguments(args)

    # Load interpreted model (TODO: generalize if using other model classes)
    model = InterpretableBertForSequenceClassification(
        model_name=args.model_dir, tokenizer_name=args.model_dir,
        batch_size=args.model_batch_size, max_seq_len=args.model_max_seq_len,
        device="cpu" if args.use_cpu else "cuda"
    )

    logging.info(f"Building sample dataset (preset_config={dataset_name}) with "
                 f"{df_sample.shape[0]} examples")
    try:
        sample_dataset = TransformerSeqDataset.build_dataset(dataset_name, df_sample,
                                                             tokenizer=model.tokenizer,
                                                             max_seq_len=args.model_max_seq_len)
    except NotImplementedError:
        sample_dataset = TransformerSeqPairDataset.build_dataset(dataset_name, df_sample,
                                                                 tokenizer=model.tokenizer,
                                                                 max_seq_len=args.model_max_seq_len)

    use_control = (args.method_class == "control")
    other_embeddings = []
    text_data = []
    # Control group is a non-overlapping group of examples from the same set as the main sample:
    # The idea is to see how well a complex model can distinguish examples from the same (empirical) distribution
    if use_control:
        control_path = os.path.join(args.experiment_dir, "control.csv")
        df_control = pd.read_csv(control_path)

        logging.info(f"Building control dataset (preset_config={dataset_name}) with "
                     f"{df_control.shape[0]} examples")
        try:
            control_dataset = TransformerSeqDataset.build_dataset(dataset_name, df_control,
                                                                  tokenizer=model.tokenizer,
                                                                  max_seq_len=args.model_max_seq_len)
        except NotImplementedError:
            control_dataset = TransformerSeqPairDataset.build_dataset(dataset_name, df_control,
                                                                      tokenizer=model.tokenizer,
                                                                      max_seq_len=args.model_max_seq_len)

        logging.info("Embedding control dataset with interpreted model")
        for curr_batch in tqdm(DataLoader(control_dataset, batch_size=args.model_batch_size)):
            other_embeddings.append(bert_embeddings(model, **curr_batch).cpu())
    else:
        # Assumption: all used models use control labels, formatted as "<LABEL_NAME>"
        possible_labels = IDX_TO_LABEL[dataset_name]
        clm_labels = [f"<{label_name.upper()}>"
                      for _, label_name in sorted(possible_labels.items(), key=lambda tup: tup[0])]
        generator = load_generator(args, clm_labels=clm_labels)
        if generator is not None:
            logging.info(f"Loaded generator ({args.generator_type})")

        nlp = stanza.Pipeline(lang="en", processors="tokenize", use_gpu=not args.use_cpu, tokenize_no_ssplit=True)
        pretokenized_test_data = [None for _ in range(df_sample.shape[0])]
        if args.custom_features is not None:
            logging.info(f"Tokenizing explained instances with Stanza")
            pretokenized_test_data = stanza_tokenize(nlp, df_sample, dataset_name)

            assert args.custom_features in ["words"], f"In distribution detection experiment, " \
                                                      f"'{args.custom_features}' is unsupported"

            if args.custom_features == "dependency_parsing":
                nlp = stanza.Pipeline(lang="en", processors="tokenize,lemma,pos,depparse")

        # Special restriction: ensure generated tokens are unique - otherwise, it is theoretically possible to achieve
        #  perfect score (~50% discrimination accuracy) by just copying the input
        if args.method_class == "ime" and generator is not None:
            logging.info("Adding unique filter to generator used in IME method")
            generator.filters = [filter_factory("unique")] + generator.filters

        used_sample_data, data_weights = None, None
        if args.method in {"ime", "ime_hybrid"}:
            logging.info(f"Loading sampling data (dataset={dataset_name})")
            df_train = load_dataset(dataset_name, file_path=args.train_path)
            df_train = df_train.sample(frac=1.0).reset_index(drop=True)

            logging.info(f"Building sampling dataset (preset_config={dataset_name}) with "
                         f"{df_train.shape[0]} examples")
            used_tokenizer = model.tokenizer if args.method == "ime" else generator.tokenizer
            used_length = args.model_max_seq_len if args.method == "ime" else args.generator_max_seq_len
            try:
                train_set = TransformerSeqDataset.build_dataset(dataset_name, df_train,
                                                                tokenizer=used_tokenizer, max_seq_len=used_length)
            except NotImplementedError:
                train_set = TransformerSeqPairDataset.build_dataset(dataset_name, df_train,
                                                                    tokenizer=used_tokenizer, max_seq_len=used_length)

            data_weights = create_uniform_weights(train_set.input_ids, train_set.special_tokens_masks)
            used_sample_data = train_set.input_ids

        logging.info("Instantiating explanation method")
        method, method_type = load_explainer(
            method_class=args.method_class, method=args.method,
            model=model, generator=generator,
            used_sample_data=used_sample_data, data_weights=data_weights,
            experiment_type="required_samples", return_generated_samples=True,
            kernel_width=1.0, shared_vocabulary=True, num_generated_samples=10
        )

        # These hold encoded perturbed samples, which will be embedded with model and used in distribution detection
        input_ids, modeling_data = [], {}

        raw_examples = df_sample[PRESET_COLNAMES[dataset_name]].values
        raw_examples = list(map(lambda row: row[0] if len(row) == 1 else tuple(row), raw_examples))

        # For each instance, sample 1 perturbation
        logging.info("Sampling perturbations")
        for raw_input, pretok_input in tqdm(zip(raw_examples, pretokenized_test_data), total=len(raw_examples)):
            is_pretok = pretok_input is not None

            # Obtain the predicted label, which the instance will be explained for
            encoded_example = model.to_internal(
                text_data=[pretok_input if is_pretok else raw_input],
                is_split_into_units=is_pretok
            )
            probas = model.score(input_ids=encoded_example["input_ids"],
                                 **{k: encoded_example["aux_data"][k] for k in ["token_type_ids", "attention_mask"]})
            predicted_label = int(torch.argmax(probas))

            feature_groups = None
            if args.custom_features is not None:
                word_ids = encoded_example["aux_data"]["alignment_ids"][0]
                # TODO: dependency_parsing
                if args.custom_features == "words":
                    feature_groups = extract_groups(word_ids)
                else:
                    raise NotImplementedError(f"Unsupported custom features: '{args.custom_features}'")

            if args.method_class == "lime":
                res = method.explain_text(raw_input, label=predicted_label,
                                          num_samples=5, explanation_length=args.explanation_length,
                                          custom_features=feature_groups)
                samples = res["samples"]
            else:
                # For IME with external LM, sampling data is generated on the fly
                if args.method == "ime_elm":
                    method.prepare_data(raw_input, label=predicted_label,
                                        pretokenized_text_data=pretok_input,
                                        custom_features=feature_groups)

                # In IME, select random sample from a random explained feature
                perturbable_indices = \
                    torch.arange(encoded_example["input_ids"].shape[1])[encoded_example["perturbable_mask"][0]]
                num_groups = perturbable_indices.shape[0] if feature_groups is None else len(feature_groups)
                if feature_groups is None:
                    selected_feature = perturbable_indices[torch.randint(num_groups, ())].item()
                else:
                    selected_feature = torch.randint(num_groups, ()).item()

                # IMPORTANT: assuming shared vocabulary, so we do not need to convert anything to generator's repr.
                res = method.estimate_feature_importance(idx_feature=selected_feature,
                                                         instance=encoded_example["input_ids"],
                                                         num_samples=2,
                                                         perturbable_mask=encoded_example["perturbable_mask"],
                                                         feature_groups=feature_groups,
                                                         **encoded_example["aux_data"])

                samples = res["samples"]

            # sample[0] is by convention the original sample in LIME, which does not use LIME's masking strategy
            idx_selected = np.random.randint(1 if args.method_class == "lime" else 0,
                                             len(samples))
            input_ids.append(samples[idx_selected])
            text_data.append(model.tokenizer.decode(input_ids[-1], skip_special_tokens=False))

            # First example: initialize lists for additional data
            if len(modeling_data.keys()) == 0:
                for k in encoded_example["aux_data"].keys():
                    modeling_data[k] = [encoded_example["aux_data"][k][0]]
            else:
                for k in modeling_data.keys():
                    modeling_data[k].append(encoded_example["aux_data"][k][0])

        input_ids = torch.tensor(input_ids)
        modeling_data = {k: torch.stack(v) for k, v in modeling_data.items()}

        num_batches = (input_ids.shape[0] + args.model_batch_size - 1) // args.model_batch_size
        for idx_batch in tqdm(range(num_batches), total=num_batches):
            s_b, e_b = idx_batch * args.model_batch_size, (idx_batch + 1) * args.model_batch_size
            other_embeddings.append(
                bert_embeddings(model, input_ids[s_b: e_b],
                                **{attr: attr_values[s_b: e_b] for attr, attr_values in modeling_data.items()}).cpu()
            )

    other_embeddings = torch.cat(other_embeddings).numpy()

    sample_embeddings = []
    for curr_batch in tqdm(DataLoader(sample_dataset, batch_size=args.model_batch_size)):
        sample_embeddings.append(bert_embeddings(model, **curr_batch).cpu())
    sample_embeddings = torch.cat(sample_embeddings).numpy()

    np.save(os.path.join(mini_experiment_path, "sample.npy"), sample_embeddings)
    np.save(os.path.join(mini_experiment_path, "other.npy"), other_embeddings)

    if len(text_data) > 0:
        with open(os.path.join(mini_experiment_path, "perturbations.txt"), "w") as f:
            f.writelines(text_data)
