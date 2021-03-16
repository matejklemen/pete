import json
import os
from typing import Union

import torch
import numpy as np
import matplotlib.pyplot as plt

from explain_nlp.experimental.arguments import parser
from explain_nlp.experimental.data import load_nli, LABEL_TO_IDX
from explain_nlp.experimental.handle_explainer import load_explainer
from explain_nlp.experimental.handle_generator import load_generator
from explain_nlp.methods.lime import LIMEMaskedLMExplainer, LIMEExplainer
from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification

parser.add_argument("--num_repeats", type=int, default=10)

if __name__ == "__main__":
    args = parser.parse_args()
    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")
    df_test = load_nli(args.test_path)

    model = InterpretableBertForSequenceClassification(tokenizer_name=args.model_dir,
                                                       model_name=args.model_dir,
                                                       batch_size=args.model_batch_size,
                                                       max_seq_len=args.model_max_seq_len,
                                                       max_words=args.model_max_words,
                                                       device="cpu" if args.use_cpu else "cuda")

    generator, generator_description = load_generator(args, clm_labels=["<ENTAILMENT>", "<NEUTRAL>", "<CONTRADICTION>"])

    assert args.method in ["lime", "lime_lm"]
    method, method_type = load_explainer(method=args.method, model=model,
                                         return_model_scores=args.return_model_scores,
                                         return_generated_samples=args.return_generated_samples,
                                         # Method-specific options below:
                                         generator=generator, num_generated_samples=args.num_generated_samples,
                                         kernel_width=args.kernel_width,
                                         # unused, but currently required by the method
                                         confidence_interval=None, max_abs_error=None)
    method = method  # type: Union[LIMEExplainer, LIMEMaskedLMExplainer]

    NUM_SAMPLES = args.min_samples_per_feature
    NUM_REPEATS = args.num_repeats
    assert args.explanation_length is not None
    EXPLANATION_LENGTH = args.explanation_length

    experiment_dir = f"{args.method}_{NUM_SAMPLES}samples_k{EXPLANATION_LENGTH}_{NUM_REPEATS}" \
        if args.experiment_dir is None else args.experiment_dir
    os.makedirs(experiment_dir)

    with open(os.path.join(experiment_dir, "experiment_config.json"), "w") as f_config:
        json.dump({
            "test_path": args.test_path,
            "method": args.method,
            "num_samples": NUM_SAMPLES,
            "explanation_length": EXPLANATION_LENGTH,
            "num_repeats": NUM_REPEATS
        }, fp=f_config, indent=4)

    selected_features_per_example = []
    for idx_example, input_pair in enumerate(df_test[["sentence1", "sentence2"]].values.tolist()):
        print(f"Processing {input_pair}")
        encoded_example = model.to_internal(text_data=[input_pair])
        probas = model.score(input_ids=encoded_example["input_ids"].to(DEVICE),
                             **{k: v.to(DEVICE) for k, v in encoded_example["aux_data"].items()})
        predicted_label = int(torch.argmax(probas))
        actual_label = int(df_test.iloc[[idx_example]]["gold_label"].apply(lambda label_str: LABEL_TO_IDX["snli"][label_str]))

        num_total_features = None

        selected_features = []
        for idx_rep in range(NUM_REPEATS):
            res = method.explain_text(text_data=input_pair, label=predicted_label,
                                      num_samples=NUM_SAMPLES, explanation_length=EXPLANATION_LENGTH)
            curr_selected = torch.flatten(torch.nonzero(res["importance"])).tolist()
            selected_features.append(curr_selected)
            num_total_features = int(res["importance"].shape[0])

        counter = np.zeros(num_total_features)
        for curr_selected in selected_features:
            counter[curr_selected] += 1

        selected_features_per_example.append(int(np.sum(counter > 0)))

        plt.title(f"Ex.#{idx_example}: Empirical selection frequency (/{NUM_REPEATS} reps) "
                  f"of K={EXPLANATION_LENGTH}-sparse LIME")
        plt.bar(np.arange(num_total_features), counter)

        plt.ylim([0, NUM_REPEATS + 1])
        plt.yticks(np.arange(0, NUM_REPEATS + 1, 10))

        plt.xlabel("Feature")
        plt.xticks(np.arange(num_total_features))
        plt.margins(x=0)

        plt.savefig(os.path.join(experiment_dir, f"ex{idx_example}.png"))
        plt.clf()

    with open(os.path.join(experiment_dir, "selected_features_per_sample.json"), "w") as f:
        json.dump({
            "selected_features_per_example": selected_features_per_example,
            "mean_selected_features": np.mean(selected_features_per_example),
            "sd_selected_features": np.std(selected_features_per_example)
        }, fp=f, indent=4)

    print(f"Selected features per example: "
          f"mean={np.mean(selected_features_per_example)},"
          f"sd={np.std(selected_features_per_example)}")
