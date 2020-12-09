import os
from typing import List, Optional, Mapping
import numpy as np


def base_visualization(body_html: str, head_js="", body_js="", path=None):
    # Obtain path to directory of this module with regard to caller path
    call_path = __file__.split(os.path.sep)[:-1]
    with open(os.path.join(os.path.sep.join(call_path), "highlight.css"), "r") as f:
        css_properties = f.read()
    with open(os.path.join(os.path.sep.join(call_path), "highlight.js"), "r") as f:
        js_code = f.read()

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
        {js_code}
        </script>
        <script>
        {body_js}
        </script>
    </body>
    </html>
    """

    if path is not None:
        with open(path, "w") as f:
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
                   path: Optional[str] = None):
    eff_actual_labels = [None for _ in range(len(sequences))] if actual_labels is None else actual_labels

    if len(sequences) != len(importances):
        raise ValueError(f"Got an unequal amount of sequences and importances of sequence elements "
                         f"({len(sequences)} sequences != {len(importances)} importances)")

    if len(sequences) != len(pred_labels):
        raise ValueError(f"Got an unequal amount of sequences and labels "
                         f"({len(sequences)} sequences != {len(pred_labels)} labels)")

    np_importances = np.array(importances)
    max_abs_imps = np.max(np.abs(np_importances), axis=1, keepdims=True)
    scaled_imps = scale_interval(np_importances, curr_low=-max_abs_imps, curr_high=max_abs_imps)

    body_html = []
    for i, (curr_seq, curr_sc_imps, curr_pred_label, curr_actual_label) in enumerate(zip(sequences,
                                                                                         scaled_imps,
                                                                                         pred_labels,
                                                                                         eff_actual_labels)):
        if len(curr_seq) != len(curr_sc_imps):
            raise ValueError(f"Example #{i}: importance not provided for each sequence element "
                             f"({len(curr_seq)} sequence elements != {len(curr_sc_imps)} )")

        curr_ex_html = []
        # If label is not given, color it with yellow, indicating "unknown correctness"
        label_color = "rgb(238, 232, 170)"
        eff_actual_label = "/"
        if curr_actual_label is not None:
            label_color = "rgba(200, 247, 197, 1)" if curr_actual_label == curr_pred_label else "rgba(236, 100, 75, 1)"
            eff_actual_label = curr_actual_label

        curr_ex_html.append(f"<span class='example-label' "
                            f"title='Actual label: {eff_actual_label}' "
                            f"style='background-color: {label_color}'> "
                            f"Predicted label: {curr_pred_label} "
                            f"</span>")
        curr_ex_html.append("<br />")

        for idx_tok, scaled_imp in enumerate(curr_sc_imps):
            # Most green and most red colors indicate highest and (-highest) importances (i.e. equal tails)
            curr_color = f"rgba(0, 153, 0, {scaled_imp})" if scaled_imp > 0 else f"rgba(255, 0, 0, {abs(scaled_imp)})"

            curr_ex_html.append(f"<span class='example-unit' "
                                f"id='token-{i}-{idx_tok}'"
                                f"title='{np_importances[i, idx_tok]: .4f}' "
                                f"style='background-color: {curr_color}'>"
                                f"{curr_seq[idx_tok]}"
                                f"</span>")

        body_html.append("<div class='example'>{}</div>".format("\n".join(curr_ex_html)))

    body_html = "\n".join(body_html)
    return base_visualization(body_html, path=path)


def highlight_plot_multiple_methods(sequences: List[List[str]],
                                    importances: Mapping[str, List[List[float]]],
                                    pred_labels: List,
                                    actual_labels: Optional[List] = None,
                                    path: Optional[str] = None):
    eff_actual_labels = [None for _ in range(len(sequences))] if actual_labels is None else actual_labels
    for method_name, method_imps in importances.items():
        if len(sequences) != len(method_imps):
            raise ValueError(f"Got an unequal amount of sequences and importances of sequence elements "
                             f"({len(sequences)} sequences != {len(method_imps)} importances for method '{method_name}')")

    if len(sequences) != len(pred_labels):
        raise ValueError(f"Got an unequal amount of sequences and labels "
                         f"({len(sequences)} sequences != {len(pred_labels)} labels)")

    np_importances = {method_name: np.array(method_imps) for method_name, method_imps in importances.items()}
    max_abs_imps = {method_name: np.max(np.abs(method_imps), axis=1, keepdims=True)
                    for method_name, method_imps in np_importances.items()}
    scaled_imps = {method_name: scale_interval(np_importances[method_name],
                                               curr_low=-max_abs_imps[method_name],
                                               curr_high=max_abs_imps[method_name])
                   for method_name in importances}

    method_order = list(importances.keys())
    body_html = []
    for i, (curr_seq, curr_pred_label, curr_actual_label) in enumerate(zip(sequences, pred_labels, eff_actual_labels)):
        curr_ex_html = []

        # If label is not given, color it with yellow, indicating "unknown correctness"
        label_color = "rgb(238, 232, 170)"
        eff_actual_label = "/"
        if curr_actual_label is not None:
            label_color = "rgba(200, 247, 197, 1)" if curr_actual_label == curr_pred_label else "rgba(236, 100, 75, 1)"
            eff_actual_label = curr_actual_label

        curr_ex_html.append(f"<span class='example-label' "
                            f"title='Actual label: {eff_actual_label}' "
                            f"style='background-color: {label_color}'> "
                            f"Predicted label: {curr_pred_label} "
                            f"</span>")
        curr_ex_html.append("<br />")

        for method_name in method_order:
            method_html = []
            method_html.append(f"<span class='method-name'>{method_name}</span>")
            method_html.append("<br />")

            curr_sc_imps = scaled_imps[method_name][i]
            if len(curr_seq) != len(curr_sc_imps):
                raise ValueError(f"Example #{i}: importance not provided for each sequence element"
                                 f"({len(curr_seq)} sequence elements != {len(curr_sc_imps)} in method '{method_name}')")

            for idx_tok, scaled_imp in enumerate(curr_sc_imps):
                # Most green and most red colors indicate highest and (-highest) importances (i.e. equal tails)
                curr_color = f"rgba(0, 153, 0, {scaled_imp})" if scaled_imp > 0 else f"rgba(255, 0, 0, {abs(scaled_imp)})"

                method_html.append(f"<span class='example-unit' "
                                   f"id='token-{i}-{idx_tok}'"
                                   f"title='{np_importances[method_name][i, idx_tok]: .4f}' "
                                   f"style='background-color: {curr_color}'>"
                                   f"{curr_seq[idx_tok]}"
                                   f"</span>")

            curr_ex_html.append("<div class='method'>{}</div>".format("\n".join(method_html)))

        body_html.append("<div class='example'>{}</div>".format("\n".join(curr_ex_html)))

    body_html = "\n".join(body_html)
    return base_visualization(body_html, path=path)


if __name__ == "__main__":
    # highlight_plot(sequences=[["The", "show", "was", "fucking", "shite", "but", "I", "liked", "the", "end"]],
    #                importances=[[0.0, -0.05, 0.0, 0.35, 0.05, 0.01, 0.0, -0.1, 0.0, -0.05]],
    #                pred_labels=["toxic"], actual_labels=["toxic"], path="tmp_refactor.html")

    from explain_nlp.experimental.core import MethodData
    ime_run1 = MethodData.load("/home/matej/Documents/embeddia/interpretability/ime-lm/experiments/SNLI/results/IME/snli_xs_accurate_values_r1/ime_data.json")
    sequences = ime_run1.sequences
    pred_labels = ime_run1.pred_labels
    actual_labels = ime_run1.actual_labels

    ime_run2 = MethodData.load(
        "/home/matej/Documents/embeddia/interpretability/ime-lm/experiments/SNLI/results/IME/snli_xs_accurate_values_r2/ime_data.json")
    ime_run3 = MethodData.load(
        "/home/matej/Documents/embeddia/interpretability/ime-lm/experiments/SNLI/results/IME/snli_xs_accurate_values_r3/ime_data.json")
    ime_run4 = MethodData.load(
        "/home/matej/Documents/embeddia/interpretability/ime-lm/experiments/SNLI/results/IME/snli_xs_accurate_values_r4/ime_data.json")
    ime_run5 = MethodData.load(
        "/home/matej/Documents/embeddia/interpretability/ime-lm/experiments/SNLI/results/IME/snli_xs_accurate_values_r5/ime_data.json")

    highlight_plot_multiple_methods(
        sequences=sequences, pred_labels=pred_labels, actual_labels=actual_labels, path="tmp_refactor.html",
        importances={
            "IME (run1)": ime_run1.importances,
            "IME (run2)": ime_run2.importances,
            "IME (run3)": ime_run3.importances,
            "IME (run4)": ime_run4.importances,
            "IME (run5)": ime_run5.importances
        }
    )
