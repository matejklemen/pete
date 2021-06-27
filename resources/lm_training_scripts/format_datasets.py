import os

from explain_nlp.experimental.data import load_sst2, load_sentinews, load_imsypp, load_nli
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


def process_sentinews(train_path, dev_path, test_path, target_dir):
    """ Converts SentiNews to LM and controlled LM format. Writes formatted data to `target_dir`. """
    train_df = load_sentinews(train_path)
    dev_df = load_sentinews(dev_path)
    test_df = load_sentinews(test_path)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Language modeling: 1 example per line, format: <sentence>
    with open(os.path.join(target_dir, "sentinews_train_lm.txt"), "w", encoding="utf-8") as f_train:
        train_df["content"].apply(lambda s: print(s, file=f_train))

    with open(os.path.join(target_dir, "sentinews_dev_lm.txt"), "w", encoding="utf-8") as f_dev:
        dev_df["content"].apply(lambda s: print(s, file=f_dev))

    with open(os.path.join(target_dir, "sentinews_test_lm.txt"), "w", encoding="utf-8") as f_test:
        test_df["content"].apply(lambda s: print(s, file=f_test))

    # Controlled language modeling: 1 example per line, format: <LABEL> <sentence>
    LABELS = {"neutral", "negative", "positive"}
    with open(os.path.join(target_dir, "sentinews_train_lm_label.txt"), "w", encoding="utf-8") as f_train:
        for seq, label in train_df[["content", "sentiment"]].values:
            assert label in LABELS
            print(f"<{label.upper()}> {seq}", file=f_train)

    with open(os.path.join(target_dir, "sentinews_dev_lm_label.txt"), "w", encoding="utf-8") as f_dev:
        for seq, label in dev_df[["content", "sentiment"]].values:
            assert label in LABELS
            print(f"<{label.upper()}> {seq}", file=f_dev)

    with open(os.path.join(target_dir, "sentinews_test_lm_label.txt"), "w", encoding="utf-8") as f_test:
        for seq, label in test_df[["content", "sentiment"]].values:
            assert label in LABELS
            print(f"<{label.upper()}> {seq}", file=f_test)


def process_imsypp(train_path, dev_path, test_path, target_dir):
    """ Converts IMSYPP-sl to LM and controlled LM format. Writes formatted data to `target_dir`. """
    train_df = load_imsypp(train_path)
    dev_df = load_imsypp(dev_path)
    test_df = load_imsypp(test_path)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Language modeling: 1 example per line, format: <sentence>
    with open(os.path.join(target_dir, "imsypp_train_lm.txt"), "w", encoding="utf-8") as f_train:
        train_df["besedilo"].apply(lambda s: print(s, file=f_train))

    with open(os.path.join(target_dir, "imsypp_dev_lm.txt"), "w", encoding="utf-8") as f_dev:
        dev_df["besedilo"].apply(lambda s: print(s, file=f_dev))

    with open(os.path.join(target_dir, "imsypp_test_lm.txt"), "w", encoding="utf-8") as f_test:
        test_df["besedilo"].apply(lambda s: print(s, file=f_test))

    # Controlled language modeling: 1 example per line, format: <LABEL> <sentence>
    LABEL_MAP = {0: "<CLEAN>", 1: "<HATE>"}
    with open(os.path.join(target_dir, "imsypp_train_lm_label.txt"), "w", encoding="utf-8") as f_train:
        for sent, label in train_df[["besedilo", "vrsta"]].values:
            print(f"{LABEL_MAP[label]} {sent}", file=f_train)

    with open(os.path.join(target_dir, "imsypp_dev_lm_label.txt"), "w", encoding="utf-8") as f_dev:
        for sent, label in dev_df[["besedilo", "vrsta"]].values:
            print(f"{LABEL_MAP[label]} {sent}", file=f_dev)

    with open(os.path.join(target_dir, "imsypp_test_lm_label.txt"), "w", encoding="utf-8") as f_test:
        for sent, label in test_df[["besedilo", "vrsta"]].values:
            print(f"{LABEL_MAP[label]} {sent}", file=f_test)


def process_xnli_multi(dev_path, test_path, target_dir):
    """ Converts XNLI to LM and controlled LM format. Writes formatted data to `target_dir`.
    `dev_path` and `test_path` are assumed to point towards multilingual data.

    Writing as sequence pairs, separated by </s></s> (i.e. for XLM-RoBERTa)
    """
    dev_df = load_nli(dev_path)
    test_df = load_nli(test_path)

    # Split dev set (15 langs) into 90%:10% train-dev split
    indices = np.random.permutation(dev_df.shape[0])
    bnd = int(0.9 * indices.shape[0])
    train_indices, dev_indices = indices[: bnd], indices[bnd:]
    train_df = dev_df.iloc[train_indices]
    dev_df = dev_df.iloc[dev_indices]

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Language modeling: 1 example per line, format: <sentence>
    with open(os.path.join(target_dir, "xnli_m_train_lm.txt"), "w", encoding="utf-8") as f_train:
        for s1, s2 in train_df[["sentence1", "sentence2"]].values.tolist():
            print(f"{s1} </s></s> {s2}", file=f_train)

    with open(os.path.join(target_dir, "xnli_m_dev_lm.txt"), "w", encoding="utf-8") as f_dev:
        for s1, s2 in dev_df[["sentence1", "sentence2"]].values.tolist():
            print(f"{s1} </s></s> {s2}", file=f_dev)

    with open(os.path.join(target_dir, "xnli_m_test_lm.txt"), "w", encoding="utf-8") as f_test:
        for s1, s2 in test_df[["sentence1", "sentence2"]].values.tolist():
            print(f"{s1} </s></s> {s2}", file=f_test)

    # "<ENTAILMENT>", "<NEUTRAL>", "<CONTRADICTION>"
    LABELS = {"entailment", "neutral", "contradiction"}
    with open(os.path.join(target_dir, "xnli_m_train_lm_label.txt"), "w", encoding="utf-8") as f_train:
        for s1, s2, label in train_df[["sentence1", "sentence2", "gold_label"]].values:
            assert label in LABELS
            print(f"<{label.upper()}> {s1} </s></s> {s2}", file=f_train)

    with open(os.path.join(target_dir, "xnli_m_dev_lm_label.txt"), "w", encoding="utf-8") as f_dev:
        for s1, s2, label in dev_df[["sentence1", "sentence2", "gold_label"]].values:
            assert label in LABELS
            print(f"<{label.upper()}> {s1} </s></s> {s2}", file=f_dev)

    with open(os.path.join(target_dir, "xnli_m_test_lm_label.txt"), "w", encoding="utf-8") as f_test:
        for s1, s2, label in test_df[["sentence1", "sentence2", "gold_label"]].values:
            assert label in LABELS
            print(f"<{label.upper()}> {s1} </s></s> {s2}", file=f_test)


if __name__ == "__main__":
    # process_sst2(train_path="/home/matej/Documents/data/SST-2/train.tsv",
    #              test_path="/home/matej/Documents/data/SST-2/dev.tsv",
    #              target_dir="/home/matej/Documents/data/SST-2/lm_data")

    # process_sentinews(train_path="/home/matej/Documents/data/sentinews/split-paragraph-level/train.txt",
    #                   dev_path="/home/matej/Documents/data/sentinews/split-paragraph-level/dev.txt",
    #                   test_path="/home/matej/Documents/data/sentinews/split-paragraph-level/test.txt",
    #                   target_dir="/home/matej/Documents/data/sentinews/split-paragraph-level/lm_data")

    # process_imsypp(train_path="/home/matej/Documents/data/imsypp/split/train.csv",
    #                dev_path="/home/matej/Documents/data/imsypp/split/dev.csv",
    #                test_path="/home/matej/Documents/data/imsypp/split/test.csv",
    #                target_dir="/home/matej/Documents/data/imsypp/split/lm_data")

    process_xnli_multi(dev_path="/home/matej/Documents/data/XNLI-1.0/xnli.dev.tsv",
                       test_path="/home/matej/Documents/data/XNLI-1.0/xnli.test.tsv",
                       target_dir="/home/matej/Documents/data/XNLI-1.0/lm_data")
