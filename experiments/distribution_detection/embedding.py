import argparse
import json
import logging
import os
import sys
from typing import Tuple, List

import numpy as np
import pandas as pd
import stanza
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from explain_nlp.experimental.arguments import log_arguments
from explain_nlp.experimental.data import TransformerSeqDataset, TransformerSeqPairDataset, load_dataset, \
    IDX_TO_LABEL, PRESET_COLNAMES
from explain_nlp.experimental.handle_generator import load_generator
from explain_nlp.experimental.handle_model import load_model
from explain_nlp.generation.decoding import filter_factory
from explain_nlp.methods.ime_lm import create_uniform_weights
from explain_nlp.methods.utils import list_indexer
from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification, \
    InterpretableXLMRobertaForSequenceClassification

"""
    These functions contain simplistic implementations of perturbations as seen in IME/LIME/their variants using LMs.
    Assumptions:
    - feature representations are same for model and generator (meaning no conversion needs to take place)
"""


def mock_ime(explained_instances: torch.Tensor, perturbable_masks: torch.Tensor,
             sample_data: torch.Tensor, feature_groups: List[List[int]] = None):
    num_instances = explained_instances.shape[0]
    num_features = explained_instances.shape[1]

    eff_feature_groups = feature_groups
    if feature_groups is None:
        eff_feature_groups = []
        for curr_mask in perturbable_masks:
            perturbable_indices = torch.arange(num_features)[curr_mask]
            eff_feature_groups.append([[_i.item()] for _i in perturbable_indices])

    perturbed_ids = []
    randomly_selected_examples = torch.randint(sample_data.shape[0], (num_instances,))
    # replace between 0 and `len(feature_groups)` feature groups /w corresponding values in randomly selected instances
    for idx_ex in range(num_instances):
        curr_example = explained_instances[idx_ex].clone()
        curr_groups = eff_feature_groups[idx_ex]
        replace_with = randomly_selected_examples[idx_ex]

        num_replaced = torch.randint(len(curr_groups), ())
        replace_groups = torch.randperm(len(curr_groups))[: num_replaced].tolist()
        replace_features = list_indexer(curr_groups, replace_groups)

        curr_example[replace_features] = sample_data[replace_with, replace_features]
        perturbed_ids.append(curr_example)

    return torch.stack(perturbed_ids)


def mock_ime_ilm(explained_instances: torch.Tensor, perturbable_masks: torch.Tensor,
                 generator, feature_groups: List[List[int]] = None,
                 **modeling_kwargs):
    num_instances = explained_instances.shape[0]
    num_features = explained_instances.shape[1]

    eff_feature_groups = feature_groups
    if feature_groups is None:
        eff_feature_groups = []
        for curr_mask in perturbable_masks:
            perturbable_indices = torch.arange(num_features)[curr_mask]
            eff_feature_groups.append([[_i.item()] for _i in perturbable_indices])

    if hasattr(generator, "label_weights"):
        randomly_selected_label = torch.multinomial(generator.label_weights,
                                                    num_samples=num_instances,
                                                    replacement=True)
        randomly_selected_label = [generator.control_labels_str[i] for i in randomly_selected_label]
    else:
        randomly_selected_label = [None] * num_instances

    is_masked = torch.zeros((num_instances, num_features), dtype=torch.bool)
    # re-generate between 0 and `len(feature_groups)` feature groups
    for idx_ex in range(num_instances):
        curr_groups = eff_feature_groups[idx_ex]

        num_masked = torch.randint(len(curr_groups), ())
        mask_groups = torch.randperm(len(curr_groups))[: num_masked].tolist()
        mask_features = list_indexer(curr_groups, mask_groups)
        is_masked[idx_ex, mask_features] = True

    _explained_instances = explained_instances.clone()
    perturbed_ids = generator.generate_masked_samples(explained_instances,
                                                      generation_mask=is_masked,
                                                      control_labels=randomly_selected_label,
                                                      **modeling_kwargs)

    assert perturbed_ids.shape[0] == _explained_instances.shape[0]

    return perturbed_ids


def mock_ime_elm(explained_instances: torch.Tensor, perturbable_masks: torch.Tensor,
                 generator, feature_groups: List[List[int]] = None,
                 **modeling_kwargs):
    # This is still a naive version because conversion is not as straightforward
    # TODO: see if I can update generate() to work with either a single ex or multiple ex
    num_instances = explained_instances.shape[0]
    num_features = explained_instances.shape[1]

    eff_feature_groups = feature_groups
    if feature_groups is None:
        eff_feature_groups = []
        for curr_mask in perturbable_masks:
            perturbable_indices = torch.arange(num_features)[curr_mask]
            eff_feature_groups.append([[_i.item()] for _i in perturbable_indices])

    if hasattr(generator, "label_weights"):
        randomly_selected_label = torch.multinomial(generator.label_weights,
                                                    num_samples=num_instances,
                                                    replacement=True)
        randomly_selected_label = [generator.control_labels_str[i] for i in randomly_selected_label]
    else:
        randomly_selected_label = [None] * num_instances

    generated_samples = []
    for idx_ex in range(num_instances):
        generator_res = generator.generate(input_ids=explained_instances[[idx_ex]],
                                           perturbable_mask=perturbable_masks[[idx_ex]],
                                           num_samples=1,
                                           control_labels=[randomly_selected_label[idx_ex]],
                                           **{k: v[[idx_ex]] for k, v in modeling_kwargs.items()})
        generated_samples.append(generator_res["input_ids"][0])

    generated_samples = torch.stack(generated_samples)
    perturbed_ids = []
    # replace between 0 and `len(feature_groups)` feature groups /w corresponding values in randomly selected instances
    for idx_ex in range(num_instances):
        curr_example = explained_instances[idx_ex].clone()
        curr_groups = eff_feature_groups[idx_ex]

        num_replaced = torch.randint(len(curr_groups), ())
        replace_groups = torch.randperm(len(curr_groups))[: num_replaced].tolist()
        replace_features = list_indexer(curr_groups, replace_groups)

        curr_example[replace_features] = generated_samples[idx_ex, replace_features]
        perturbed_ids.append(curr_example)

    return torch.stack(perturbed_ids)


def mock_lime(explained_instances: torch.Tensor, perturbable_masks: torch.Tensor, replace_with_token: int,
              feature_groups: List[List[int]] = None):
    num_instances = explained_instances.shape[0]
    num_features = explained_instances.shape[1]

    eff_feature_groups = feature_groups
    if feature_groups is None:
        eff_feature_groups = []
        for curr_mask in perturbable_masks:
            perturbable_indices = torch.arange(num_features)[curr_mask]
            eff_feature_groups.append([[_i.item()] for _i in perturbable_indices])

    perturbed_ids = []
    # replace between 0 and `len(feature_groups)` feature groups /w corresponding values in randomly selected instances
    for idx_ex in range(num_instances):
        curr_example = explained_instances[idx_ex].clone()
        curr_groups = eff_feature_groups[idx_ex]

        num_replaced = torch.randint(len(curr_groups), ())
        replace_groups = torch.randperm(len(curr_groups))[: num_replaced].tolist()
        replace_features = list_indexer(curr_groups, replace_groups)

        curr_example[replace_features] = replace_with_token
        perturbed_ids.append(curr_example)

    return torch.stack(perturbed_ids)


""" 
    This is copypasted and trimmed from arguments in arguments.py in order to minimize redundant arguments that are 
    stored with the experiments (it becomes really unclear what is actually being used)
"""
general_parser = argparse.ArgumentParser(add_help=False)
general_parser.add_argument("--experiment_dir", type=str, default="debug_snli")
general_parser.add_argument("--mini_experiment_name", type=str, default=None,
                            help="Use a custom name for this mini-experiment. By default, experiments are named after "
                                 "used methods, but this can be problematic if using same method with different params")
general_parser.add_argument("--random_seed", type=int, default=None)
general_parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of GPU")
general_parser.add_argument("--custom_features", type=str, default=None,
                            choices=[None, "words", "dependency_parsing"])
general_parser.add_argument("--model_dir", type=str,
                            default="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/classifiers/snli_bert_uncased")
general_parser.add_argument("--model_type", type=str, default="bert_sequence")
general_parser.add_argument("--model_max_seq_len", type=int, default=41)
general_parser.add_argument("--model_batch_size", type=int, default=8)

general_parser.add_argument("--generator_type", type=str, default="bert_mlm",
                            choices=["bert_mlm", "bert_cmlm"])
general_parser.add_argument("--generator_dir", type=str,
                            default="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/language_models/bert-base-uncased-snli-mlm")
general_parser.add_argument("--generator_batch_size", type=int, default=8)
general_parser.add_argument("--generator_max_seq_len", type=int, default=41)
general_parser.add_argument("--strategy", type=str, default="top_p",
                            choices=["top_k", "top_p"])
general_parser.add_argument("--top_p", type=float, default=0.0001)  # = greedy by default
general_parser.add_argument("--top_k", type=int, default=3)
general_parser.add_argument("--threshold", type=float, default=0.1)
general_parser.add_argument("--unique_dropout", type=float, default=0.0)

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


@torch.no_grad()
def xlm_roberta_embeddings(model: InterpretableXLMRobertaForSequenceClassification, input_ids, **modeling_kwargs):
    # take <s> token, passed through another linear layer:
    # see forward function of XLM-R/RoBERTa seq. classifier in `transformers`
    outputs = model.model.roberta(input_ids=input_ids.to(model.device),
                                  **{attr: modeling_kwargs[attr].to(model.device)
                                     for attr in ["attention_mask"]})
    zeroeth_token = outputs[0][:, 0, :]  # [num_examples, hidden_size]

    return torch.tanh(model.model.classifier.dense(zeroeth_token))


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

    if dataset_name == "xnli":
        # Sort examples by language to minimize the number of times the Stanza tokenizer is reloaded for a different lg
        df_sample = df_sample.sort_values("language").reset_index(drop=True)

    if args.mini_experiment_name is not None:
        mini_experiment_path = os.path.join(args.experiment_dir, args.mini_experiment_name)
    else:
        mini_experiment_path = os.path.join(args.experiment_dir, args.method)

    logging.info(f"Saving mini-experiment results to '{mini_experiment_path}'")
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

    model = load_model(model_type=args.model_type, model_name=args.model_dir, tokenizer_name=args.model_dir,
                       batch_size=args.model_batch_size, max_seq_len=args.model_max_seq_len,
                       device="cpu" if args.use_cpu else "cuda")
    if args.model_type == "bert_sequence":
        model_embeddings = bert_embeddings
    elif args.model_type == "xlmr_sequence":
        model_embeddings = xlm_roberta_embeddings
    else:
        raise NotImplementedError(f"Unsupported model type in embeddings.py: '{args.model_type}'")

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
            other_embeddings.append(model_embeddings(model, **curr_batch).cpu())
    else:
        # Assumption: all used models use control labels, formatted as "<LABEL_NAME>"
        possible_labels = IDX_TO_LABEL[dataset_name]
        clm_labels = [f"<{label_name.upper()}>"
                      for _, label_name in sorted(possible_labels.items(), key=lambda tup: tup[0])]
        generator = load_generator(args, clm_labels=clm_labels)
        if generator is not None:
            logging.info(f"Loaded generator ({args.generator_type})")

        # TODO: Add support for units bigger than subwords
        if args.custom_features is not None:
            logging.info(f"Tokenizing explained instances with Stanza")
            used_stanza_lang = "sl" if dataset_name in ["sentinews", "imsypp"] else "en"
            nlp = stanza.Pipeline(lang=used_stanza_lang, processors="tokenize", tokenize_no_ssplit=True,
                                  use_gpu=not args.use_cpu)

            if dataset_name == "xnli":
                # TODO: at least Thai and Swahili currently do not have Stanza tokenizers, so they will need fallbacks
                pretokenized_test_data = []
                for lang, group in df_sample.groupby("language"):
                    pretokenized_test_data.extend(
                        stanza_tokenize(stanza.Pipeline(lang, processors="tokenize"), group, dataset_name)
                    )
            else:
                pretokenized_test_data = stanza_tokenize(nlp, df_sample, dataset_name)

            assert args.custom_features in ["words"], f"In distribution detection experiment, " \
                                                      f"'{args.custom_features}' is unsupported"

            if args.custom_features == "dependency_parsing":
                # TODO: XNLI will need to handle this differently
                nlp = stanza.Pipeline(lang=used_stanza_lang, processors="tokenize,lemma,pos,depparse")

        # Special restriction: ensure generated tokens are unique - otherwise, it is theoretically possible to achieve
        #  perfect score (~50% discrimination accuracy) by just copying the input
        if args.method_class == "ime" and generator is not None:
            logging.info("Adding unique filter to generator used in IME method")
            generator.filters = [filter_factory("unique")] + generator.filters

        raw_examples = df_sample[PRESET_COLNAMES[dataset_name]].values
        raw_examples = list(map(lambda row: row[0] if len(row) == 1 else tuple(row), raw_examples))

        encoded_sample = model.to_internal(raw_examples)
        modeling_data = encoded_sample["aux_data"]

        if args.method == "ime":
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

            used_sample_data = train_set.input_ids
            data_weights = create_uniform_weights(train_set.input_ids, train_set.special_tokens_mask)

            # XNLI: handle each language on its own in order to use sampling data only from that language
            if dataset_name == "xnli":
                xnli_lang_to_indices = {}
                for lang, group in df_train.groupby("language"):
                    xnli_lang_to_indices[lang] = group.index.tolist()

                input_ids, modeling_data = [], {}
                for curr_lang, curr_group in df_sample.groupby("language"):
                    logging.info(f"[XNLI] lg='{curr_lang}', {curr_group.shape[0]} examples")
                    encoded_sample = model.to_internal(list(zip(curr_group["sentence1"].tolist(),
                                                                curr_group["sentence2"].tolist())))
                    input_ids.append(mock_ime(encoded_sample["input_ids"], encoded_sample["perturbable_mask"],
                                              sample_data=used_sample_data[xnli_lang_to_indices[curr_lang]],
                                              feature_groups=None))
                    for k, v in encoded_sample["aux_data"].items():
                        if k not in modeling_data:
                            modeling_data[k] = [v]
                        else:
                            modeling_data[k].append(v)

                input_ids = torch.cat(input_ids)
                modeling_data = {k: torch.cat(v) for k, v in modeling_data.items()}
            else:
                input_ids = mock_ime(encoded_sample["input_ids"], encoded_sample["perturbable_mask"],
                                     sample_data=used_sample_data, feature_groups=None)
        elif args.method == "lime":
            input_ids = mock_lime(encoded_sample["input_ids"], encoded_sample["perturbable_mask"],
                                  replace_with_token=model.pad_token_id, feature_groups=None)
        elif args.method == "ime_elm":
            input_ids = mock_ime_elm(encoded_sample["input_ids"], encoded_sample["perturbable_mask"],
                                     generator, feature_groups=None, **encoded_sample["aux_data"])
        # LIME + LM and IME + internal LM use generators in the same way
        elif args.method in ["lime_lm", "ime_ilm"]:
            input_ids = mock_ime_ilm(encoded_sample["input_ids"], encoded_sample["perturbable_mask"],
                                     generator, feature_groups=None, **encoded_sample["aux_data"])
        else:
            raise NotImplementedError(f"Invalid method '{args.method}'")

        assert input_ids.shape[0] == len(raw_examples)
        for _i in range(len(raw_examples)):
            text_data.append(model.tokenizer.decode(input_ids[_i]))

        num_batches = (input_ids.shape[0] + args.model_batch_size - 1) // args.model_batch_size
        for idx_batch in tqdm(range(num_batches), total=num_batches):
            s_b, e_b = idx_batch * args.model_batch_size, (idx_batch + 1) * args.model_batch_size
            other_embeddings.append(
                model_embeddings(model, input_ids[s_b: e_b],
                                 **{attr: attr_values[s_b: e_b] for attr, attr_values in modeling_data.items()}).cpu()
            )

    other_embeddings = torch.cat(other_embeddings).numpy()

    sample_embeddings = []
    for curr_batch in tqdm(DataLoader(sample_dataset, batch_size=args.model_batch_size)):
        sample_embeddings.append(model_embeddings(model, **curr_batch).cpu())
    sample_embeddings = torch.cat(sample_embeddings).numpy()

    np.save(os.path.join(mini_experiment_path, "sample.npy"), sample_embeddings)
    np.save(os.path.join(mini_experiment_path, "other.npy"), other_embeddings)

    if len(text_data) > 0:
        with open(os.path.join(mini_experiment_path, "perturbations.txt"), "w") as f:
            for line in text_data:
                print(line, file=f)
