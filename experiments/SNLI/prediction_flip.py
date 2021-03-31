import itertools
import json
import logging
import os
import sys
from time import time
from typing import List

import stanza
import torch

from explain_nlp.experimental.arguments import methods_parser, runtime_parse_args, log_arguments
from explain_nlp.experimental.data import load_nli, TransformerSeqPairDataset, LABEL_TO_IDX, IDX_TO_LABEL
from explain_nlp.experimental.handle_explainer import load_explainer
from explain_nlp.experimental.handle_features import handle_features
from explain_nlp.experimental.handle_generator import load_generator
from explain_nlp.methods.hybrid import create_uniform_weights
from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification
from explain_nlp.visualizations.highlight import highlight_plot

if __name__ == "__main__":
    args = methods_parser.parse_args()
    args = runtime_parse_args(args)
    if not os.path.exists(os.path.join(args.experiment_dir, "explanations")):
        os.makedirs(os.path.join(args.experiment_dir, "explanations"))

    # Set up logging to file and stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout),
                         logging.FileHandler(os.path.join(args.experiment_dir, "experiment.log"))]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)

    log_arguments(args)

    nlp = stanza.Pipeline(lang="en", processors="tokenize", use_gpu=(not args.use_cpu), tokenize_no_ssplit=True)
    pretokenized_test_data = None

    model = InterpretableBertForSequenceClassification(
        model_name=args.model_dir,
        tokenizer_name=args.model_dir,
        batch_size=args.model_batch_size,
        max_seq_len=args.model_max_seq_len,
        device=("cpu" if args.use_cpu else "cuda")
    )
    model_description = {"type": "bert", "max_seq_len": args.model_max_seq_len, "handle": args.model_dir}
    generator, gen_description = load_generator(args, clm_labels=["<ENTAILMENT>", "<NEUTRAL>", "<CONTRADICTION>"])

    df_test = load_nli(args.test_path)
    num_examples = df_test.shape[0]
    if args.custom_features is not None:
        pretokenized_test_data = []
        for idx_subset in range((df_test.shape[0] + 1024 - 1) // 1024):
            s, e = idx_subset * 1024, (1 + idx_subset) * 1024
            for s0, s1 in zip(nlp("\n\n".join(df_test["sentence1"].iloc[s: e].values)).sentences,
                              nlp("\n\n".join(df_test["sentence2"].iloc[s: e].values)).sentences):
                pretokenized_test_data.append((
                    [token.words[0].text for token in s0.tokens],
                    [token.words[0].text for token in s1.tokens]
                ))

    used_data = {"test_path": args.test_path}
    used_sample_data = None
    data_weights = None
    if args.method in {"ime", "ime_hybrid"}:
        logging.info("Loading training dataset as sampling data")
        used_data["train_path"] = args.train_path
        df_train = load_nli(args.train_path).sample(frac=1.0).reset_index(drop=True)
        used_tokenizer = model.tokenizer if args.method == "ime" else generator.tokenizer
        train_set = TransformerSeqPairDataset.build(df_train["sentence1"].tolist(), df_train["sentence2"].tolist(),
                                                    labels=df_train["gold_label"].apply(
                                                        lambda label_str: LABEL_TO_IDX["snli"][label_str]).tolist(),
                                                    tokenizer=used_tokenizer, max_seq_len=args.model_max_seq_len)
        data_weights = create_uniform_weights(train_set.input_ids, train_set.special_tokens_masks)
        used_sample_data = train_set.input_ids

    method, method_type = load_explainer(model=model, generator=generator,
                                         used_sample_data=used_sample_data, data_weights=data_weights,
                                         **vars(args))
    num_taken_samples = args.min_samples_per_feature if args.method_class == "ime" else args.num_samples

    if args.custom_features is not None:
        # Reload pipeline if depparse features are used (not done from the start as this would slow down tokenization)
        if args.custom_features.startswith("depparse"):
            nlp = stanza.Pipeline(lang="en", processors="tokenize,lemma,pos,depparse")
        else:
            nlp = stanza.Pipeline(lang="en", processors="tokenize")

    results = {
        "idx_example": [],
        "num_removals_to_flip": [], "num_possible_removals": [], "frac_removals_to_flip": [],
        "removed": [],
        "label_before": [], "proba_before": [],
        "label_after": [], "proba_after": [],
        "aggregate": {}
    }

    if os.path.exists(os.path.join(args.experiment_dir, "experiment_data.json")):
        with open(os.path.join(args.experiment_dir, "experiment_data.json"), "r", encoding="utf-8") as f:
            results = json.load(f)

    start_from = min(int(args.start_from) if args.start_from is not None else len(results["idx_example"]), num_examples)
    until = min(int(args.until) if args.until is not None else num_examples, num_examples)

    logging.info(f"Running computation from example#{start_from} (inclusive) to example#{until} (exclusive)")
    for idx_example in range(start_from, until):
        curr_example = df_test.iloc[idx_example]
        input_pair = tuple(curr_example[["sentence1", "sentence2"]])

        encoded = model.to_internal(
            text_data=[pretokenized_test_data[idx_example] if args.custom_features is not None else input_pair],
            is_split_into_units=args.custom_features is not None
        )
        original_proba = model.score(input_ids=encoded["input_ids"],
                                     **{k: v for k, v in encoded["aux_data"].items()
                                        if k in ["token_type_ids", "attention_mask"]})
        original_label = int(torch.argmax(original_proba[0]))

        pretokenized_example, curr_features = None, None
        if args.custom_features is not None:
            # Obtain word IDs for subwords in all cases as the custom features are usually obtained from words
            word_ids = encoded["aux_data"]["alignment_ids"][0]
            curr_features = handle_features(args.custom_features,
                                            word_ids=word_ids,
                                            raw_example=tuple(input_pair),
                                            pipe=nlp)
            pretokenized_example = pretokenized_test_data[idx_example]

        side_results = {}
        t1 = time()
        if args.method_class == "lime":
            res = method.explain_text(input_pair, pretokenized_text_data=pretokenized_example,
                                      label=original_label, num_samples=num_taken_samples,
                                      explanation_length=args.explanation_length,
                                      custom_features=curr_features)
        else:
            raise NotImplementedError("Method IME is not handled yet!")

        t2 = time()
        logging.info(f"[{args.method}] Example {idx_example}: time taken = {t2 - t1:.2f}s")

        sequence_tokens = res["input"]
        num_importances = res["importance"].shape[0]

        # primary features
        if num_importances == model.max_seq_len:
            importance_indices = torch.arange(model.max_seq_len)[encoded["perturbable_mask"][0]]
            features = [[int(idx_feature)] for idx_feature in importance_indices]
        # custom features: noted after placholder importances for primary features
        elif num_importances > model.max_seq_len:
            importance_indices = torch.arange(model.max_seq_len, num_importances)
            features = res["custom_features"]  # type: List[List[int]]
        else:
            raise ValueError(f"Weird: num_importances={num_importances}<max_seq_len={model.max_seq_len}")

        valid_importances = res["importance"][importance_indices]
        desc_ordering = torch.argsort(-valid_importances)
        asc_ordering = torch.argsort(valid_importances)

        # worst case: have to remove all features and the decision still doesn't flip
        num_removals_to_flip, removed = int(valid_importances.shape[0]), None
        flipped_label, flipped_proba = None, None

        for (curr_ordering, order_type) in zip([desc_ordering, asc_ordering],
                                               ["descending", "ascending"]):

            for num_removed in range(1, 1 + valid_importances.shape[0]):
                curr_removed = list(itertools.chain(*[features[_i] for _i in curr_ordering[:num_removed]]))
                keep_mask = torch.ones_like(encoded["input_ids"], dtype=torch.bool)
                keep_mask[0, curr_removed] = False

                partial_input_ids = encoded["input_ids"][keep_mask].unsqueeze(0)
                partial_aux_data = {k: encoded["aux_data"][k][keep_mask].unsqueeze(0)
                                    for k in ["token_type_ids", "attention_mask"]}

                proba = model.score(partial_input_ids, **partial_aux_data)
                new_label = int(torch.argmax(proba[0]))

                if new_label != original_label:
                    logging.info(f"[{order_type} order] Prediction flipped after removing {num_removed} features!")
                    if num_removed <= num_removals_to_flip:
                        num_removals_to_flip = num_removed
                        removed = curr_removed
                        flipped_label = new_label
                        flipped_proba = float(proba[0, original_label])

                    break

        logging.info(f"[FINAL] Removals needed: {num_removals_to_flip}")
        results["idx_example"].append(idx_example)
        results["num_removals_to_flip"].append(num_removals_to_flip)
        results["num_possible_removals"].append(int(valid_importances.shape[0]))
        results["frac_removals_to_flip"].append(num_removals_to_flip / int(valid_importances.shape[0]))
        results["removed"].append(removed)
        results["label_before"].append(original_label)
        results["proba_before"].append(float(original_proba[0, original_label]))
        results["label_after"].append(flipped_label)
        results["proba_after"].append(flipped_proba)

        results["aggregate"] = {
            "frac_removals_to_flip": {
                "mean": float(torch.mean(torch.tensor(results["frac_removals_to_flip"]))),
                "sd": float(torch.std(torch.tensor(results["frac_removals_to_flip"])))
            },
            "label_after": {k: v.tolist() for k, v in zip(["label", "count"],
                                                          torch.unique(torch.tensor(results["label_after"]),
                                                                       return_counts=True))}
        }

        highlight_plot([res["input"]], importances=[res["importance"].tolist()],
                       pred_labels=[IDX_TO_LABEL["snli"][original_label]],
                       actual_labels=[curr_example["gold_label"]],
                       custom_features=(res["custom_features"] if args.custom_features is not None else None),
                       path=os.path.join(args.experiment_dir, "explanations", f"ex{str(idx_example).zfill(4)}.html"))

        with open(os.path.join(args.experiment_dir, "experiment_data.json"), "w", encoding="utf-8") as f:
            json.dump(results, fp=f, indent=4)
