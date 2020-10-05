import pandas as pd
import os


def extract_dataset(data_dir):
    """ Gather up reviews scattered across files into a single structure. """
    reviews, labels = [], []
    for idx_label, label in enumerate(["neg", "pos"]):
        label_dir = os.path.join(data_dir, label)
        files = [file_name for file_name in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, file_name))]

        for curr_file in files:
            with open(os.path.join(label_dir, curr_file), encoding="utf-8") as f:
                reviews.append(f.read())
                labels.append(idx_label)

    return reviews, labels


if __name__ == "__main__":
    # The original dataset is scattered around 50k files: gather them up and create an intermediate file whose loading
    # will be much faster
    train_dir = "/home/matej/Documents/data/aclImdb/train"
    test_dir = "/home/matej/Documents/data/aclImdb/test"
    train_reviews, train_labels = extract_dataset(train_dir)
    pd.DataFrame({"review": train_reviews, "label": train_labels}).to_csv(os.path.join(train_dir, "data.csv"),
                                                                          index=False)
    test_reviews, test_labels = extract_dataset(test_dir)
    pd.DataFrame({"review": test_reviews, "label": test_labels}).to_csv(os.path.join(test_dir, "data.csv"),
                                                                        index=False)
