import os

from explain_nlp.experimental.data import load_sst2
import numpy as np


def process_sst2(train_path, test_path, target_dir):
    """ Converts SST2 to LM and controlled LM format. Writes formatted data to `target_dir`. """
    train_df = load_sst2(train_path)
    test_df = load_sst2(test_path)

    indices = np.random.permutation(train_df.shape[0])
    bnd = int(0.9 * train_df.shape[0])
    train_inds = indices[:bnd]
    dev_inds = indices[bnd:]

    dev_df = train_df.iloc[dev_inds]
    train_df = train_df.iloc[train_inds]

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Language modeling: 1 example per line, format: <sentence>
    train_df["sentence"].to_csv(os.path.join(target_dir, "sst2_train_lm.txt"), index=False, header=False)
    dev_df["sentence"].to_csv(os.path.join(target_dir, "sst2_dev_lm.txt"), index=False, header=False)
    test_df["sentence"].to_csv(os.path.join(target_dir, "sst2_test_lm.txt"), index=False, header=False)

    # Controlled language modeling: 1 example per line, format: <LABEL> <sentence>
    LABEL_MAP = {0: "<NEGATIVE>", 1: "<POSITIVE>"}
    with open(os.path.join(target_dir, "sst2_train_lm_label.txt"), "w", encoding="utf-8") as f_train:
        for sent, label in train_df[["sentence", "label"]].values:
            print(f"{LABEL_MAP[label]} {sent}", file=f_train)

    with open(os.path.join(target_dir, "sst2_dev_lm_label.txt"), "w", encoding="utf-8") as f_dev:
        for sent, label in dev_df[["sentence", "label"]].values:
            print(f"{LABEL_MAP[label]} {sent}", file=f_dev)

    with open(os.path.join(target_dir, "sst2_test_lm_label.txt"), "w", encoding="utf-8") as f_test:
        for sent, label in test_df[["sentence", "label"]].values:
            print(f"{LABEL_MAP[label]} {sent}", file=f_test)


if __name__ == "__main__":
    process_sst2(train_path="/home/matej/Documents/data/SST-2/train.tsv",
                 test_path="/home/matej/Documents/data/SST-2/dev.tsv",
                 target_dir="/home/matej/Documents/data/SST-2/lm_data")
