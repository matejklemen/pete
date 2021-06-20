import os

from explain_nlp.experimental.data import load_nli
from argparse import ArgumentParser
import logging

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="snli",
                    choices=["snli"])
parser.add_argument("--data_path", type=str, default="/home/matej/Documents/data/snli/snli_1.0_test.txt")
parser.add_argument("--experiment_dir", type=str, default="debug")
parser.add_argument("--random_seed", type=int, default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logging.info("Extracting samples:")
    for attr, val in vars(args).items():
        logging.info(f"\t- {attr}={val}")

    if args.dataset == "snli":
        df = load_nli(args.data_path)
    else:
        raise NotImplementedError(f"'{args.dataset}'")

    num_samples = df.shape[0]
    df_shuffled = df.sample(frac=1.0, random_state=args.random_seed).reset_index(drop=True)

    subset1 = df_shuffled.iloc[:num_samples//2]
    subset2 = df_shuffled.iloc[num_samples//2:]

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    subset1.to_csv(os.path.join(args.experiment_dir, "sample.csv"), index=False)
    subset2.to_csv(os.path.join(args.experiment_dir, "control.csv"), index=False)
