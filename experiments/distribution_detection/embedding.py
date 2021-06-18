import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from explain_nlp.experimental.data import TransformerSeqPairDataset, LABEL_TO_IDX
from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification

parser = ArgumentParser()
parser.add_argument("--experiment_dir", type=str,
                    default="debug")
parser.add_argument("--method", type=str,
                    choices=["control", "ime", "lime", "ime_ilm", "ime_elm", "lime_lm"],
                    default="control")

parser.add_argument("--model_dir", type=str,
                    default="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased")
parser.add_argument("--model_max_seq_len", type=int,
                    default=41)
parser.add_argument("--model_batch_size", type=int,
                    default=16)

parser.add_argument("--random_seed", type=int, default=None)
parser.add_argument("--use_cpu", action="store_true", default=True)


@torch.no_grad()
def bert_embeddings(model: InterpretableBertForSequenceClassification, input_ids, **modeling_kwargs):
    # BERT: pooler_output -> dropout -> linear -> class
    output = model.model.bert(input_ids=input_ids.to(model.device),
                              **{attr: modeling_kwargs[attr].to(model.device)
                                 for attr in ["token_type_ids", "attention_mask"]})
    return output["pooler_output"]  # [num_examples, hidden_size]


if __name__ == "__main__":
    args = parser.parse_args()
    assert os.path.exists(args.experiment_dir), \
        "--experiment_dir must point to a valid directory. Please run sample.py first in order to create it"

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    # TODO: save config of mini-experiment
    # ...

    # Load sample data
    sample_path = os.path.join(args.experiment_dir, "sample.csv")
    assert os.path.exists(sample_path)
    df_sample = pd.read_csv(sample_path)

    use_control = (args.method == "control")

    mini_experiment_path = os.path.join(args.experiment_dir, args.method)
    if not os.path.exists(mini_experiment_path):
        os.makedirs(mini_experiment_path)

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

    sample_embeddings = []
    for curr_batch in tqdm(DataLoader(sample_dataset, batch_size=args.model_batch_size)):
        sample_embeddings.append(bert_embeddings(model, **curr_batch).cpu())
    sample_embeddings = torch.cat(sample_embeddings).numpy()

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
        # TODO: if no_control_data():
        #   TODO: load generator
        #   ...

        #   TODO: load explainer
        #   ...

        #   TODO: for each instance, sample 1 perturbation
        #   ...

        raise NotImplementedError

    other_embeddings = torch.cat(other_embeddings).numpy()

    np.save(os.path.join(mini_experiment_path, "sample.npy"), sample_embeddings)
    np.save(os.path.join(mini_experiment_path, "other.npy"), other_embeddings)
