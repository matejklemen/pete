import csv
import json
import logging
import os
import sys
from ast import literal_eval
from typing import Union, List, Optional, Tuple

import pandas as pd
import stanza
import torch
import numpy as np

from explain_nlp.experimental.arguments import methods_parser, runtime_parse_args, log_arguments
from explain_nlp.experimental.handle_explainer import load_explainer
from explain_nlp.experimental.handle_features import handle_features
from explain_nlp.experimental.handle_generator import load_generator
from explain_nlp.modeling.modeling_transformers import InterpretableRobertaForSequenceClassification
from explain_nlp.utils.metrics import iou_score, hpd_score, snr_score
from explain_nlp.visualizations.highlight import highlight_plot


def load_squadv2_quackie(path, sample_size=None):
    df = pd.read_csv(path, sep=",", quoting=csv.QUOTE_ALL, nrows=sample_size)
    df["ground_truth_sents"] = df["ground_truth_sents"].apply(literal_eval)

    return df


if __name__ == "__main__":
    # TODO: enable IME as well
    args = methods_parser.parse_args()
    args = runtime_parse_args(args)

    if not os.path.exists(os.path.join(args.experiment_dir, "explanations")):
        os.makedirs(os.path.join(args.experiment_dir, "explanations"))

    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")
    data = load_squadv2_quackie(args.test_path)
    num_examples = data.shape[0]
    nlp = stanza.Pipeline("en", processors="tokenize", use_gpu=(not args.use_cpu))

    # Set up logging to file and stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout),
                         logging.FileHandler(os.path.join(args.experiment_dir, "experiment.log"))]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)

    log_arguments(args)

    # Make the tokens belonging to question unperturbable as we are not interested in importance of question
    def custom_to_internal(self, text_data: Union[List[str], List[Tuple[str, ...]],
                                                  List[List[str]], List[Tuple[List[str], ...]]],
                           is_split_into_units: Optional[bool] = False,
                           allow_truncation: Optional[bool] = True):
        truncation_strategy = "longest_first" if allow_truncation else "do_not_truncate"
        _encoded = self.encode_aligned(text_data,
                                       is_split_into_units=is_split_into_units,
                                       truncation_strategy=truncation_strategy)
        for _i in range(len(text_data)):
            sep_indices = torch.flatten(torch.nonzero(_encoded["input_ids"][_i] == self.tokenizer.sep_token_id,
                                                      as_tuple=False))
            _encoded["perturbable_mask"][_i, sep_indices[0]:] = False

        return _encoded

    InterpretableRobertaForSequenceClassification.to_internal = custom_to_internal
    model = InterpretableRobertaForSequenceClassification(
        model_name="a-ware/roberta-large-squad-classification",
        tokenizer_name="a-ware/roberta-large-squad-classification",
        batch_size=args.model_batch_size,
        max_seq_len=512,
        device="cpu" if args.use_cpu else "cuda"
    )

    generator, generator_description = load_generator(args)

    args_dict = vars(args)
    args_dict["method_class"] = "lime"
    method, method_type = load_explainer(model=model, generator=generator,
                                         **args_dict)

    pred_sents, gt_sents = [], []
    snr, hpd, iou = [], [], []
    results = {
        "id": [],
        "pred_sent": [],
        "gt_sent": [],
        "snr": [],
        "hpd": [],
        "iou": [],
        "aggregate": {}
    }

    if os.path.exists(os.path.join(args.experiment_dir, "experiment_data.json")):
        with open(os.path.join(args.experiment_dir, "experiment_data.json"), "r", encoding="utf-8") as f:
            results = json.load(f)

    start_from = min(int(args.start_from) if args.start_from is not None else len(results["id"]), num_examples)
    until = min(int(args.until) if args.until is not None else num_examples, num_examples)

    logging.info(f"Running computation from example#{start_from} (inclusive) to example#{until} (exclusive)")
    for idx_ex in range(start_from, until):
        logging.info(f"Example#{idx_ex}...")
        curr_example = data.iloc[idx_ex]
        results["id"].append(curr_example["id"])

        context = curr_example["context"]
        question = curr_example["question"]
        gt_sents = curr_example["ground_truth_sents"]

        example_words = [[], []]
        for idx_sent, sent in enumerate(nlp(context).sentences):
            example_words[0].extend([w.text for w in sent.words])

        for idx_sent, sent in enumerate(nlp(question).sentences):
            example_words[1].extend([w.text for w in sent.words])

        example_words = tuple(example_words)

        encoded_example = model.to_internal(text_data=[example_words], is_split_into_units=True)
        word_ids = encoded_example["aux_data"]["alignment_ids"][0]
        sent_features = handle_features("sentences", word_ids=word_ids,
                                        raw_example=(context, question),
                                        pipe=nlp)
        # Last sentence is the question (if tokenized properly), which we don't want the importance for
        num_sents = len(sent_features) - 1
        sent_features = sent_features[:-1]

        curr_features = sent_features # TODO: if args.aggregation_strategy == "sentence" else word_ids

        # We know the label is positive because the preprocessing script only keeps answers that are both
        # answerable as per ground truth annotation and as per model prediction
        predicted_label, actual_label = 1, 1
        res = method.explain_text(text_data=example_words, label=predicted_label,
                                  pretokenized_text_data=example_words, custom_features=curr_features,
                                  num_samples=args.num_samples, explanation_length=args.explanation_length)
        sent_importances = res["importance"][-num_sents:].numpy()
        ordering = np.argsort(-sent_importances)

        pred_binary = np.zeros(num_sents, dtype=np.int32)
        pred_binary[ordering[0]] = 1

        idx_best_gt = np.argmax([iou_score(y_true=np.eye(1, num_sents, k=curr_gt, dtype=np.int32)[0],
                                           y_pred=pred_binary) for curr_gt in gt_sents])
        best_gt = np.eye(1, num_sents, k=gt_sents[idx_best_gt], dtype=np.int32)[0]

        results["pred_sent"].append(int(ordering[0]))
        results["gt_sent"].append(int(idx_best_gt))

        results["iou"].append(iou_score(best_gt, pred_binary))
        results["hpd"].append(hpd_score(ordering, gt=best_gt))
        results["snr"].append(snr_score(sent_importances, gt=best_gt))

        highlight_plot([res["input"]], importances=[res["importance"].tolist()],
                       pred_labels=["answerable"],
                       actual_labels=["answerable"],
                       custom_features=[curr_features],
                       path=os.path.join(args.experiment_dir, "explanations", f"ex{str(idx_ex).zfill(4)}.html"))

        if len(results["id"]) >= 2:
            results["aggregate"] = {
                "iou": {"mean": np.mean(results["iou"]), "sd": np.std(results["iou"])},
                "hpd": {"mean": np.mean(results["hpd"]), "sd": np.std(results["hpd"])},
                "snr": {"mean": np.mean(results["snr"]), "sd": np.std(results["snr"])}
            }

        with open(os.path.join(args.experiment_dir, "experiment_data.json"), "w", encoding="utf-8") as f:
            logging.info("Saving experiment data...")
            json.dump(results, f, indent=4)


