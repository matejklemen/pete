import json
import logging
import os
import sys
from time import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from explain_nlp.experimental.arguments import lime_parser
from explain_nlp.experimental.data import load_nli, LABEL_TO_IDX
from explain_nlp.experimental.handle_explainer import load_explainer
from explain_nlp.experimental.handle_generator import load_generator
from explain_nlp.methods.lime import LIMEMaskedLMExplainer, LIMEExplainer
from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification

lime_parser.add_argument("--num_repeats", type=int, default=10)

if __name__ == "__main__":
    args = lime_parser.parse_args()
    assert args.explanation_length is not None
    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")

    experiment_dir = f"{args.method}_{args.num_samples}samples_k{args.explanation_length}_{args.num_repeats}reps" \
        if args.experiment_dir is None else args.experiment_dir

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Set up logging to file and stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout),
                         logging.FileHandler(os.path.join(experiment_dir, "experiment.log"))]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)

    # Display used experiment settings on stdout and save to file
    for k, v in vars(args).items():
        v_str = str(v)
        v_str = f"...{v_str[-(50-3):]}" if len(v_str) > 50 else v_str
        logging.info(f"|{k:30s}|{v_str:50s}|")

    model = InterpretableBertForSequenceClassification(tokenizer_name=args.model_dir,
                                                       model_name=args.model_dir,
                                                       batch_size=args.model_batch_size,
                                                       max_seq_len=args.model_max_seq_len,
                                                       max_words=args.model_max_words,
                                                       device="cpu" if args.use_cpu else "cuda")

    generator, generator_description = load_generator(args, clm_labels=["<ENTAILMENT>", "<NEUTRAL>", "<CONTRADICTION>"])
    args_dict = vars(args)
    args_dict["method_class"] = "lime"
    method, method_type = load_explainer(model=model, generator=generator,
                                         **args_dict)  # type: Union[LIMEExplainer, LIMEMaskedLMExplainer]

    with open(os.path.join(experiment_dir, "experiment_config.json"), "w") as f_config:
        json.dump(vars(args), fp=f_config, indent=4)

    selected_features_per_example = []
    start_from = 0
    if os.path.exists(os.path.join(experiment_dir, "selected_features_per_sample.json")):
        with open(os.path.join(experiment_dir, "selected_features_per_sample.json"), "r", encoding="utf-8") as f:
            existing_data = json.load(f)

        selected_features_per_example = existing_data["selected_features_per_example"]
        start_from = len(selected_features_per_example)
        logging.info(f"Loaded existing experiment data - continuing from example#{start_from}")

    df_test = load_nli(args.test_path)
    for idx_example, input_pair in enumerate(df_test[["sentence1", "sentence2"]].values[start_from:].tolist(),
                                             start=start_from):
        logging.info(f"#{idx_example}. Processing {input_pair}")
        encoded_example = model.to_internal(text_data=[input_pair])
        probas = model.score(input_ids=encoded_example["input_ids"].to(DEVICE),
                             **{k: v.to(DEVICE) for k, v in encoded_example["aux_data"].items()})
        predicted_label = int(torch.argmax(probas))
        actual_label = int(df_test.iloc[[idx_example]]["gold_label"].apply(lambda label_str: LABEL_TO_IDX["snli"][label_str]))

        num_total_features = None

        selected_features = []
        for idx_rep in range(args.num_repeats):
            ts = time()
            res = method.explain_text(text_data=input_pair, label=predicted_label,
                                      num_samples=args.num_samples, explanation_length=args.explanation_length)
            curr_selected = torch.flatten(torch.nonzero(res["importance"])).tolist()
            selected_features.append(curr_selected)
            num_total_features = int(res["importance"].shape[0])
            te = time()
            logging.info(f"\t[Rep#{idx_rep}] Time taken: {te - ts: .4f}s")

        counter = np.zeros(num_total_features)
        for curr_selected in selected_features:
            counter[curr_selected] += 1

        selected_features_per_example.append(int(np.sum(counter > 0)))

        plt.title(f"Ex.#{idx_example}: Empirical selection frequency (/{args.num_repeats} reps) "
                  f"of K={args.explanation_length}-sparse LIME")
        plt.bar(np.arange(num_total_features), counter)

        plt.ylim([0, args.num_repeats + 1])
        plt.yticks(np.arange(0, args.num_repeats + 1, 10))

        plt.xlabel("Feature")
        plt.xticks(np.arange(num_total_features))
        plt.margins(x=0)

        plt.savefig(os.path.join(experiment_dir, f"ex{idx_example}.png"))
        plt.clf()

        logging.info(f"Saving updated data")
        with open(os.path.join(experiment_dir, "selected_features_per_sample.json"), "w", encoding="utf-8") as f:
            json.dump({
                "selected_features_per_example": selected_features_per_example,
                "mean_selected_features": np.mean(selected_features_per_example),
                "sd_selected_features": np.std(selected_features_per_example)
            }, fp=f, indent=4)

    logging.info(f"[FINAL RESULTS] Selected features per example: "
                 f"mean={np.mean(selected_features_per_example)}, "
                 f"sd={np.std(selected_features_per_example)}")
