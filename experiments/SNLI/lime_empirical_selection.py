import json
import logging
import os
import sys
from time import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import stanza
import torch

from explain_nlp.experimental.arguments import lime_parser
from explain_nlp.experimental.data import load_nli, LABEL_TO_IDX, IDX_TO_LABEL
from explain_nlp.experimental.handle_explainer import load_explainer
from explain_nlp.experimental.handle_features import handle_features
from explain_nlp.experimental.handle_generator import load_generator
from explain_nlp.methods.lime import LIMEMaskedLMExplainer, LIMEExplainer
from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification
from explain_nlp.utils.metrics import fidelity
from explain_nlp.visualizations.highlight import highlight_plot

lime_parser.add_argument("--num_repeats", type=int, default=10)
lime_parser.add_argument("--shuffle_generation_order", action="store_true",
                         help="Whether to increase generation variance by shuffling the order "
                              "(if the generator allows it)")

if __name__ == "__main__":
    args = lime_parser.parse_args()
    assert args.explanation_length is not None
    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")

    experiment_dir = f"{args.method}_{args.num_samples}samples_k{args.explanation_length}_{args.num_repeats}reps" \
        if args.experiment_dir is None else args.experiment_dir

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        os.makedirs(os.path.join(experiment_dir, "plots"))
        os.makedirs(os.path.join(experiment_dir, "explanations"))

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

    experiment_data = []
    start_from = 0
    if os.path.exists(os.path.join(experiment_dir, "experiment_data.json")):
        with open(os.path.join(experiment_dir, "experiment_data.json"), "r", encoding="utf-8") as f:
            existing_data = json.load(f)

        experiment_data = existing_data["examples"]
        start_from = len(experiment_data)
        logging.info(f"Loaded existing experiment data - continuing from example#{start_from}")

    df_test = load_nli(args.test_path)

    nlp, pretokenized_test_data = None, None
    if args.custom_features is not None:
        nlp = stanza.Pipeline(lang="en", processors="tokenize", use_gpu=not args.use_cpu, tokenize_no_ssplit=True)
        pretokenized_test_data = []
        for idx_subset in range((df_test.shape[0] + 1024 - 1) // 1024):
            s, e = idx_subset * 1024, (1 + idx_subset) * 1024
            for s0, s1 in zip(nlp("\n\n".join(df_test["sentence1"].iloc[s: e].values)).sentences,
                              nlp("\n\n".join(df_test["sentence2"].iloc[s: e].values)).sentences):
                pretokenized_test_data.append((
                    [token.words[0].text for token in s0.tokens],
                    [token.words[0].text for token in s1.tokens]
                ))

        # Reload pipeline if depparse features are used (not done from the start as this would slow down tokenization)
        if args.custom_features.startswith("depparse"):
            nlp = stanza.Pipeline(lang="en", processors="tokenize,lemma,pos,depparse")

    for idx_example, input_pair in enumerate(df_test[["sentence1", "sentence2"]].values[start_from:].tolist(),
                                             start=start_from):
        logging.info(f"#{idx_example}. Processing {input_pair}")

        if args.custom_features is not None:
            encoded_example = model.to_internal(pretokenized_text_data=[pretokenized_test_data[idx_example]])
        else:
            encoded_example = model.to_internal(text_data=[input_pair])

        probas = model.score(input_ids=encoded_example["input_ids"].to(DEVICE),
                             **{k: v.to(DEVICE) for k, v in encoded_example["aux_data"].items()})
        predicted_label = int(torch.argmax(probas))
        actual_label = int(LABEL_TO_IDX["snli"][df_test.iloc[idx_example]["gold_label"]])

        pretokenized_example, curr_features = None, None
        if args.custom_features is not None:
            # Obtain word IDs for subwords in all cases as the custom features are usually obtained from words
            word_ids = encoded_example["aux_data"]["alignment_ids"][0].tolist()
            curr_features = handle_features(args.custom_features, word_ids=word_ids, pipe=nlp,
                                            raw_example=(df_test.iloc[idx_example]["sentence1"],
                                                         df_test.iloc[idx_example]["sentence2"]))
            pretokenized_example = pretokenized_test_data[idx_example]

        num_total_features = None
        reps_data = []

        # Track importances of best and worst (according to fidelity) explanations
        best_fidelity, worst_fidelity = -float("inf"), float("inf")
        best_res, worst_res = None, None
        for idx_rep in range(args.num_repeats):
            ts = time()
            res = method.explain_text(text_data=input_pair, label=predicted_label,
                                      pretokenized_text_data=pretokenized_example, custom_features=curr_features,
                                      num_samples=args.num_samples, explanation_length=args.explanation_length)
            te = time()
            logging.info(f"\n[Rep#{idx_rep}] (time taken={te - ts: .2f}s)")
            curr_selected = torch.flatten(torch.nonzero(res["importance"], as_tuple=False)).tolist()
            num_total_features = int(res["importance"].shape[0])

            curr_surrogate_fidelity = fidelity(model_scores=res["pred_model"], surrogate_scores=res["pred_surrogate"])
            if curr_surrogate_fidelity < worst_fidelity:
                worst_fidelity = curr_surrogate_fidelity
                worst_res = res
            if curr_surrogate_fidelity > best_fidelity:
                best_fidelity = curr_surrogate_fidelity
                best_res = res

            reps_data.append({
                "selected_features": curr_selected,
                "fidelities": {
                    "surrogate": curr_surrogate_fidelity,
                    "mean_regressor": fidelity(model_scores=res["pred_model"], surrogate_scores=res["pred_mean"]),
                    "median_regressor": fidelity(model_scores=res["pred_model"], surrogate_scores=res["pred_median"])
                },
                "time_taken": te - ts
            })

        # Store visualization of best and worst (according to fidelity) explanations
        highlight_plot([best_res["input"]], importances=[best_res["importance"].tolist()],
                       pred_labels=[IDX_TO_LABEL["snli"][predicted_label]],
                       actual_labels=[IDX_TO_LABEL["snli"][actual_label]],
                       custom_features=[curr_features] if args.custom_features is not None else None,
                       path=os.path.join(experiment_dir, "explanations", f"ex{idx_example}_best_explanation.html"))
        highlight_plot([worst_res["input"]], importances=[worst_res["importance"].tolist()],
                       pred_labels=[IDX_TO_LABEL["snli"][predicted_label]],
                       actual_labels=[IDX_TO_LABEL["snli"][actual_label]],
                       custom_features=[curr_features] if args.custom_features is not None else None,
                       path=os.path.join(experiment_dir, "explanations", f"ex{idx_example}_worst_explanation.html"))

        counter = np.zeros(num_total_features)
        for curr_rep in reps_data:
            counter[curr_rep["selected_features"]] += 1

        example_data = {
            "reps": reps_data,
            "aggregate": {
                "fidelities": {
                    model_key: {
                        "mean": np.mean([curr_rep['fidelities'][model_key] for curr_rep in reps_data]),
                        "sd": np.std([curr_rep['fidelities'][model_key] for curr_rep in reps_data]),
                        "best": np.max([curr_rep['fidelities'][model_key] for curr_rep in reps_data]),
                        "worst": np.min([curr_rep['fidelities'][model_key] for curr_rep in reps_data])
                    } for model_key in ["surrogate", "mean_regressor", "median_regressor"]
                },
                "num_unique_selected_features": int(np.sum(counter > 0))
            }
        }
        experiment_data.append(example_data)
        logging.info("Aggregate: ")
        logging.info(example_data["aggregate"])

        plt.title(f"Ex.#{idx_example}: Empirical selection frequency (/{args.num_repeats} reps) "
                  f"of K={args.explanation_length}-sparse LIME")
        plt.bar(np.arange(num_total_features), counter)

        plt.ylim([0, args.num_repeats + 1])
        plt.yticks(np.arange(0, args.num_repeats + 1, 10))

        plt.xlabel("Feature")
        plt.xticks(np.arange(num_total_features))
        plt.margins(x=0)

        plt.savefig(os.path.join(experiment_dir, "plots", f"ex{idx_example}_num_selected.png"))
        plt.clf()

        logging.info(f"Saving updated data")
        with open(os.path.join(experiment_dir, "experiment_data.json"), "w", encoding="utf-8") as f:
            json.dump({
                "examples": experiment_data,
                "aggregate": {
                    "fidelities": {
                        model_key: {
                            "mean": np.mean([curr_ex['aggregate']['fidelities'][model_key]['mean'] for curr_ex in experiment_data]),
                            "sd": np.std([curr_ex['aggregate']['fidelities'][model_key]['mean'] for curr_ex in experiment_data])
                        } for model_key in ["surrogate", "mean_regressor", "median_regressor"]
                    },
                    "num_unique_selected_features": {
                        "mean": np.mean([curr_ex['aggregate']['num_unique_selected_features'] for curr_ex in experiment_data]),
                        "sd": np.std([curr_ex['aggregate']['num_unique_selected_features'] for curr_ex in experiment_data])
                    }
                }
            }, fp=f, indent=4)
