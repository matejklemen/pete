import json
import os
import numpy as np
import pandas as pd
from explain_nlp.experimental.arguments import methods_parser, subparsers, general_parser

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from explain_nlp.experimental.data import TransformerSeqPairDataset, LABEL_TO_IDX, load_nli
from explain_nlp.experimental.handle_explainer import load_explainer
from explain_nlp.experimental.handle_generator import load_generator
from explain_nlp.generation.decoding import filter_factory
from explain_nlp.methods.ime_lm import create_uniform_weights
from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification

control_parser = subparsers.add_parser("control", parents=[general_parser])
control_parser.add_argument("--method", choices=["control"],
                            default="control", help="Fixed argument that is here just for consistency")


@torch.no_grad()
def bert_embeddings(model: InterpretableBertForSequenceClassification, input_ids, **modeling_kwargs):
    # BERT: pooler_output -> dropout -> linear -> class
    output = model.model.bert(input_ids=input_ids.to(model.device),
                              **{attr: modeling_kwargs[attr].to(model.device)
                                 for attr in ["token_type_ids", "attention_mask"]})
    return output["pooler_output"]  # [num_examples, hidden_size]


if __name__ == "__main__":
    args = methods_parser.parse_args()

    assert os.path.exists(args.experiment_dir), \
        "--experiment_dir must point to a valid directory. Please run sample.py first in order to create it"

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    # Load sample data
    sample_path = os.path.join(args.experiment_dir, "sample.csv")
    assert os.path.exists(sample_path)
    df_sample = pd.read_csv(sample_path)

    use_control = (args.method_class == "control")

    mini_experiment_path = os.path.join(args.experiment_dir, args.method)
    if not os.path.exists(mini_experiment_path):
        os.makedirs(mini_experiment_path)

    with open(os.path.join(mini_experiment_path, "embedding_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), fp=f, indent=4)

    # Load interpreted model
    # TODO: generalize
    model = InterpretableBertForSequenceClassification(
        model_name=args.model_dir, tokenizer_name=args.model_dir,
        batch_size=args.model_batch_size, max_seq_len=args.model_max_seq_len,
        device="cpu" if args.use_cpu else "cuda"
    )
    model.model.eval()

    # TODO: generalize
    sample_dataset = TransformerSeqPairDataset.build(
        first=df_sample["sentence1"].tolist(), second=df_sample["sentence2"].tolist(),
        labels=df_sample["gold_label"].apply(lambda label_str: LABEL_TO_IDX["snli"][label_str]).tolist(),
        tokenizer=model.tokenizer, max_seq_len=args.model_max_seq_len
    )

    other_embeddings = []
    # Control group is a non-overlapping group of examples from the same set as the main sample:
    # The idea is to see how well a complex model can distinguish examples from the same (empirical) distribution
    if use_control:
        control_path = os.path.join(args.experiment_dir, "control.csv")
        df_control = pd.read_csv(control_path)

        # TODO: generalize
        control_dataset = TransformerSeqPairDataset.build(
            first=df_control["sentence1"].tolist(), second=df_control["sentence2"].tolist(),
            labels=df_control["gold_label"].apply(lambda label_str: LABEL_TO_IDX["snli"][label_str]).tolist(),
            tokenizer=model.tokenizer, max_seq_len=args.model_max_seq_len
        )

        for curr_batch in tqdm(DataLoader(control_dataset, batch_size=args.model_batch_size)):
            other_embeddings.append(bert_embeddings(model, **curr_batch).cpu())
    else:
        generator, gen_description = load_generator(args,
                                                    clm_labels=["<ENTAILMENT>", "<NEUTRAL>", "<CONTRADICTION>"])

        if args.method_class == "ime" and generator is not None:
            generator.filters = [filter_factory("unique")] + generator.filters

        used_sample_data = None
        data_weights = None
        if args.method in {"ime", "ime_hybrid"}:
            df_train = load_nli(args.train_path).sample(frac=1.0).reset_index(drop=True)  # TODO: generalize
            used_tokenizer = model.tokenizer if args.method == "ime" else generator.tokenizer
            # TODO: generalize
            train_set = TransformerSeqPairDataset.build(df_train["sentence1"].tolist(), df_train["sentence2"].tolist(),
                                                        labels=df_train["gold_label"].apply(
                                                            lambda label_str: LABEL_TO_IDX["snli"][label_str]).tolist(),
                                                        tokenizer=used_tokenizer, max_seq_len=args.model_max_seq_len)
            data_weights = create_uniform_weights(train_set.input_ids, train_set.special_tokens_masks)
            used_sample_data = train_set.input_ids

        #   TODO: load explainer (all args..., set them to reasonable defaults)
        method, method_type = load_explainer(
            method_class=args.method_class, method=args.method, model=model, generator=generator,
            used_sample_data=used_sample_data, data_weights=data_weights, experiment_type="required_samples",
            return_generated_samples=True, kernel_width=1.0, shared_vocabulary=True, num_generated_samples=10
        )

        # These hold encoded perturbed samples, which will be embedded with model and used in distribution detection
        input_ids, modeling_data = [], {}

        # For each instance, sample 1 perturbation
        for idx_ex in tqdm(range(df_sample.shape[0]), total=df_sample.shape[0]):
            example = df_sample.iloc[idx_ex]
            tup_input_pair = (example["sentence1"], example["sentence2"])

            # TODO: support for pretokenized
            encoded_example = model.to_internal(
                text_data=[tup_input_pair],
                is_split_into_units=False
            )

            probas = model.score(input_ids=encoded_example["input_ids"],
                                 **{k: encoded_example["aux_data"][k] for k in ["token_type_ids", "attention_mask"]})
            predicted_label = int(torch.argmax(probas))

            if args.method_class == "lime":
                res = method.explain_text(tup_input_pair, label=predicted_label,
                                          num_samples=5, explanation_length=args.explanation_length)
                samples = res["samples"]
            else:
                # In IME, select random sample from a random explained feature
                perturbable_indices = \
                    torch.arange(encoded_example["input_ids"].shape[1])[encoded_example["perturbable_mask"][0]]
                selected_feature = perturbable_indices[torch.randint(perturbable_indices.shape[0], ())].item()

                res = method.explain_text(tup_input_pair, label=predicted_label,
                                          min_samples_per_feature=2)
                samples = res["samples"][selected_feature]

            # sample[0] is by convention the original sample in LIME
            idx_selected = np.random.randint(1 if args.method_class == "lime" else 0,
                                             len(samples))
            input_ids.append(samples[idx_selected])

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
