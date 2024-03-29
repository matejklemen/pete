import os
from itertools import groupby
from typing import List, Optional, Mapping
import numpy as np
from explain_nlp.experimental.core import MethodData
from explain_nlp.methods.utils import incremental_mean, incremental_var
import html


def base_visualization(body_html: str, head_js="", body_js="", path=None):
    # Obtain path to directory of this module with regard to caller path
    call_path = __file__.split(os.path.sep)[:-1]
    with open(os.path.join(os.path.sep.join(call_path), "highlight.css"), "r") as f:
        css_properties = f.read()

    visualization = \
    f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style type="text/css">
        {css_properties}
        </style>
        <script type="text/javascript">
        {head_js}
        </script>
    </head>
    <body>
        <div class="examples-container">
            <div class="examples">
            {body_html}
            </div>
        </div>
        <script type="text/javascript">
        {body_js}
        </script>
    </body>
    </html>
    """

    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            print(f"Stored visualization into '{path}'")
            f.write(visualization)

    return visualization


def scale_interval(values: np.array, curr_low: float, curr_high: float,
                   new_low: Optional[float] = -1.0, new_high: Optional[float] = 1.0):
    """ Scale [curr_low, curr_high] to [new_low, new_high] linearly. """
    return (new_high - new_low) * (values - curr_low) / (curr_high - curr_low) + new_low


def highlight_plot(sequences: List[List[str]],
                   importances: List[List[float]],
                   pred_labels: List,
                   actual_labels: Optional[List] = None,
                   custom_features: Optional[List[List[List[int]]]] = None,  # list of feature groups per example
                   path: Optional[str] = None,
                   skip_tokens: List[str] = None):
    eff_skip_tokens = {"[PAD]"} if skip_tokens is None else set(skip_tokens)
    eff_actual_labels = [None for _ in range(len(sequences))] if actual_labels is None else actual_labels
    eff_custom_features = [[] for _ in range(len(sequences))] if custom_features is None else custom_features

    if len(sequences) != len(importances):
        raise ValueError(f"Got an unequal amount of sequences and importances of sequence elements "
                         f"({len(sequences)} sequences != {len(importances)} importances)")

    if len(sequences) != len(pred_labels):
        raise ValueError(f"Got an unequal amount of sequences and labels "
                         f"({len(sequences)} sequences != {len(pred_labels)} labels)")

    if len(sequences) != len(eff_custom_features):
        raise ValueError(f"Got an unequal amount of sequences and custom feature groups "
                         f"({len(sequences)} sequences != feature groups for {len(eff_custom_features)} examples)")

    np_importances, scaled_imps = [], []
    for i, curr_imps in enumerate(importances):
        # If custom features are provided, only use those importances for normalization
        idx_start_importance = len(sequences[i]) if len(eff_custom_features[i]) > 0 else 0
        curr_np = np.array(curr_imps)
        np_importances.append(curr_np)
        max_abs_imp = np.max(np.abs(curr_np[idx_start_importance:]))
        scaled_imps.append(scale_interval(curr_np, curr_low=-max_abs_imp, curr_high=max_abs_imp))

    body_html = []
    for i, (curr_seq, curr_pred_label, curr_actual_label, curr_custom_features) in enumerate(zip(sequences,
                                                                                                 pred_labels,
                                                                                                 eff_actual_labels,
                                                                                                 eff_custom_features)):
        curr_sc_imps = scaled_imps[i]
        if len(curr_seq) + len(curr_custom_features) != len(curr_sc_imps):
            raise ValueError(f"Example #{i}: importance not provided for each feature (group) "
                             f"({len(curr_seq) + len(curr_custom_features)} features != "
                             f"{len(curr_sc_imps)} importances)")

        relevant_features = curr_custom_features if len(curr_custom_features) > 0 else [[i] for i in range(len(curr_seq))]
        is_covered = np.zeros(len(curr_seq))
        for curr_group in relevant_features:
            if np.any(is_covered[curr_group]):
                raise ValueError(f"Features are not allowed to have multiple importances (feature group {curr_group} "
                                 f"overlaps with some other group)")

        idx_start_importance = len(curr_seq) if len(curr_custom_features) > 0 else 0
        # Some tokens may be grouped and therefore share the importance, stored at a custom index
        token_to_importance = np.arange(len(curr_seq))
        is_start = np.zeros(len(curr_seq), dtype=bool)
        is_end = np.zeros(len(curr_seq), dtype=bool)
        is_multiunit = np.zeros(len(curr_seq), dtype=bool)

        for idx_importance, curr_group in enumerate(relevant_features, start=idx_start_importance):
            # Feature groups might not be contiguous, so find out the beginnings and endings of their parts
            # e.g. group [1, 2, 7, 8] will be visualized as spans [1, 2] and [7, 8]
            for _, g in groupby(enumerate(sorted(curr_group)), lambda x: x[0] - x[1]):
                cont_features = list(map(lambda tup: tup[1], g))
                is_start[cont_features[0]] = True
                is_end[cont_features[-1]] = True
                if len(curr_group) > 1:
                    is_multiunit[cont_features[0]] = True

            for curr_feature in curr_group:
                token_to_importance[curr_feature] = idx_importance

        curr_ex_html = []
        # If label is not given, color it with yellow, indicating "unknown correctness"
        label_correctness_color = "rgb(252, 221, 136)"
        eff_actual_label = "/"
        if curr_actual_label is not None:
            label_correctness_color = "rgb(141, 199, 105)" if curr_actual_label == curr_pred_label \
                else "rgb(255, 176, 176)"
            eff_actual_label = curr_actual_label

        curr_ex_html.append(f"<div class='example-label' "
                            f"title='Actual label: {eff_actual_label}'> "
                            f"Explanation for: "
                            f"<div class='square' style='background-color: {label_correctness_color}'></div>"
                            f"<strong> {curr_pred_label}</strong>"
                            f"</div>")

        is_open = False
        for idx_tok, curr_tok in enumerate(curr_seq):
            if curr_tok in eff_skip_tokens:
                continue

            unscaled_imp = np_importances[i][token_to_importance[idx_tok]]
            scaled_imp = curr_sc_imps[token_to_importance[idx_tok]]
            # Most green and most red colors indicate highest and (-highest) importances (i.e. equal tails)
            if -1e-5 < unscaled_imp < 1e-5:
                highlight_color = "rgb(255, 255, 255)"
                outline_color = "rgb(255, 255, 255)"
            else:
                highlight_color = f"rgba(135, 220, 61, {scaled_imp})" if scaled_imp > 0 else f"rgba(212, 68, 68, {abs(scaled_imp)})"
                outline_color = f"rgb(55, 136, 5)" if scaled_imp > 0 else f"rgb(204, 0, 0)"

            tok_parts = []
            if is_start[idx_tok]:
                tok_parts.append(f"<div style='display: inline-flex; flex-direction: column;'>")
                tok_parts.append(f"<span class='row' style='font-size: 8pt; padding: 0; margin: 0; text-align: center; line-height: 10px'>({unscaled_imp:.3f})</span>")
                tok_parts.append(f"<span class='example-unit' "
                                 f"title='{unscaled_imp:.4f}' "
                                 f"style='background-color: {highlight_color}; border-width: 2px; border-style: solid; border-color: {outline_color}' "
                                 f"data-sequence='{i}' data-feature='{token_to_importance[idx_tok]}'>")
                # Display an ID in front of (possibly scattered) units making up large, discontiguous unit
                if is_multiunit[idx_tok]:
                    tok_parts.append(f"<span class='unit-id'"
                                     f"style='background-color: {outline_color}; border-width: 2px; border-style: solid; border-color: {outline_color}; user-select: none;'>"
                                     f"#{token_to_importance[idx_tok]}"
                                     f"</span>")

                is_open = True

            if is_open:
                tok_parts.append(f" {html.escape(str(curr_tok))}")
            else:
                # Hack to handle units that are not provided in custom features and also not skipped
                # (e.g., special tokens like [CLS])
                tok_parts.append(f"<div style='display: inline-flex; flex-direction: column;'>")
                tok_parts.append(f"<span class='row' style='font-size: 8pt; padding: 0; margin: 0; text-align: center; line-height: 10px'>({0.000:.3f})</span>")
                tok_parts.append(f"<span class='example-unit' "
                                 f"title='{0.0000:.4f}' "
                                 f"style='background-color: rgb(255, 255, 255); border-width: 2px; border-style: solid; border-color: rgb(255, 255, 255)' "
                                 f"data-sequence='{i}' data-feature='0.0'>  {html.escape(str(curr_tok))} </span>"
                                 f"</span></div>")

            if is_end[idx_tok]:
                tok_parts.append(f"</span>")
                tok_parts.append(f"</span></div>")
                is_open = False

            curr_ex_html.append("".join(tok_parts))

        body_html.append("<div class='example' id='{}'>{}</div>".format(i, "\n".join(curr_ex_html)))

    body_html = "\n".join(body_html)
    return base_visualization(body_html, path=path)


if __name__ == "__main__":
    highlight_plot(sequences=[["The", "big", "brown", "fox", "jumps", "over", "a", "lazy", "dog"]],
                   importances=[[0.02, 0.2, 0.1, 0.1, 0.05, 0.05, 0.01, -0.1, 0.1,
                                 0.001, 0.2, 0.1, 0.05, 0.05, -0.1]],
                   pred_labels=["clean"],
                   actual_labels=["clean"],
                   custom_features=[[[0], [1, 2, 3, 7], [4], [5], [6], [8]]],
                   path="tmp_refactor.html")
