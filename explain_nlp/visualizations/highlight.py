import os
from typing import List, Optional, Mapping
import numpy as np
from explain_nlp.experimental.core import MethodData


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


def track_progress(method_data: MethodData, idx_example: int,
                   path: Optional[str] = None, track_every_n_steps=100):  # TODO: determine what we actually need
    """ Reconstructing the construction of explanation. In a nutshell, this is a simplistic reimplementation of IME
    with added HTML construction. """
    sequence = method_data.sequences[idx_example]
    predicted_label = method_data.pred_labels[idx_example]
    actual_label = method_data.actual_labels[idx_example]
    min_samples_per_feature = method_data.min_samples_per_feature
    taken_samples = method_data.num_samples[idx_example]
    samples = method_data.samples[idx_example]
    scores = method_data.model_scores[idx_example]

    example_html = []
    # If label is not given, color it with yellow, indicating "unknown correctness"
    label_color = "rgb(238, 232, 170)"
    eff_actual_label = "/"
    if actual_label is not None:
        label_color = "rgba(200, 247, 197, 1)" if actual_label == predicted_label else "rgba(236, 100, 75, 1)"
        eff_actual_label = method_data.possible_labels[actual_label]

    example_html.append(f"<span class='example-label' "
                        f"title='Actual label: {eff_actual_label}' "
                        f"style='background-color: {label_color}'> "
                        f"Predicted label: {method_data.possible_labels[predicted_label]} "
                        f"</span>")
    example_html.append("<br />")

    np_scores = [np.array(curr_scores) for curr_scores in scores]

    # Step 0 = initial estimation, then each step means drawing one additional sample
    idx_step = 0
    num_features = len(sequence)
    importance_means = np.zeros(num_features, dtype=np.float32)
    importance_vars = np.zeros(num_features, dtype=np.float32)

    perturbable_mask = np.array([curr_taken > 0 for curr_taken in taken_samples])
    perturbable_features = np.arange(num_features)[perturbable_mask]

    # Track how many samples have been accounted for (1 sample = [1 score with feature, 1 score_without])
    accounted_samples = np.zeros(num_features, dtype=np.int32)
    accounted_samples[np.logical_not(perturbable_mask)] = 1
    # Initial pass, estimate variance of features using `min_samples_per_feature` features
    for idx_substep, idx_feature in enumerate(perturbable_features):
        step_html = []
        step_html.append(f"<span class='step-details'>"
                         f"{idx_step}.{idx_substep} Initial estimation for feature {idx_feature}"
                         f"</span>")
        step_html.append(f"<span title='Toggle display of samples' "
                         f"style='cursor: pointer;' "
                         f"onclick='toggleDisplay(\"step-{idx_step}-{idx_substep}-samples\")'>"
                         f"[...]"
                         f"</span>")
        step_html.append("<br />")

        curr_cursor = accounted_samples[idx_feature] * 2
        # `idx_feature` fixed
        curr_scores_with = np_scores[idx_feature][curr_cursor: curr_cursor + (min_samples_per_feature * 2): 2, predicted_label]
        # `idx_feature` randomized (possibly same as in original instance)
        curr_scores_without = np_scores[idx_feature][curr_cursor + 1: curr_cursor + (min_samples_per_feature * 2): 2, predicted_label]

        diffs = curr_scores_with - curr_scores_without
        importance_means[idx_feature] = np.mean(diffs)
        importance_vars[idx_feature] = np.var(diffs)
        accounted_samples[idx_feature] += min_samples_per_feature

        samples_html = []
        for idx_sample in range(curr_cursor, curr_cursor + (min_samples_per_feature * 2)):
            curr_label_score = np_scores[idx_feature][idx_sample, predicted_label]

            if idx_sample % 2 == 0:
                curr_pair_diff = np.abs(np_scores[idx_feature][idx_sample, predicted_label] -
                                        np_scores[idx_feature][idx_sample + 1, predicted_label])
                samples_html.append(f"<div class='pair' "
                                    f"style='background-color: rgba(255, 106, 0, {curr_pair_diff})'>")

            samples_html.append(f"<div class='pair-example' "
                                f"title='P(&#375; = {predicted_label}) = {curr_label_score:.4f}'>"
                                f"{'+' if idx_sample % 2 == 0 else '-'} "
                                f"{samples[idx_feature][idx_sample]}"
                                f"</div>")

            if idx_sample % 2 == 1:
                samples_html.append("</div>")

        step_html.append("<div id='step-{}-{}-samples' style='display: none;'>{}</div>".format(idx_step, idx_substep, "\n".join(samples_html)))

        max_abs_imps = np.max(np.abs(importance_means))
        scaled_imps = scale_interval(importance_means, curr_low=-max_abs_imps, curr_high=max_abs_imps)

        for idx_tok, scaled_imp in enumerate(scaled_imps):
            # Most green and most red colors indicate highest and (-highest) importances (i.e. equal tails)
            curr_color = f"rgba(0, 153, 0, {scaled_imp})" if scaled_imp > 0 else f"rgba(255, 0, 0, {abs(scaled_imp)})"
            imp_sd = f"{np.sqrt(importance_vars[idx_tok] / accounted_samples[idx_tok]): .4f}" \
                if accounted_samples[idx_tok] > 0 else "N/A"

            step_html.append(f"<span class='example-unit' "
                             f"id='token-0-{idx_tok}'"
                             f"title='{importance_means[idx_tok]: .4f} &#177; {imp_sd}' "
                             f"style='background-color: {curr_color}'>"
                             f"{sequence[idx_tok]}"
                             f"</span>")

        example_html.append("<div id='step-{}-{}'>{}</div>".format(idx_step, idx_substep, "\n".join(step_html)))

    idx_step += 1
    samples_counter = np.sum(accounted_samples[perturbable_mask])
    total_samples = np.sum(taken_samples)
    while samples_counter < total_samples:
        var_diffs = (importance_vars / accounted_samples) - (importance_vars / (accounted_samples + 1))
        chosen_feature = int(np.argmax(var_diffs))
        _cursor = accounted_samples[chosen_feature] * 2

        if idx_step % track_every_n_steps == 0:
            step_html = []
            step_html.append(f"<span class='step-details'>"
                             f"{idx_step}. Taking additional sample for feature {chosen_feature}"
                             f"</span>")
            step_html.append(f"<span title='Toggle display of samples' "
                             f"style='cursor: pointer;' "
                             f"onclick='toggleDisplay(\"step-{idx_step}-samples\")'>"
                             f"[...]"
                             f"</span>")
            step_html.append("<br />")

            samples_html = []
            for idx_sample in range(_cursor, _cursor + 2):
                curr_label_score = np_scores[chosen_feature][idx_sample, predicted_label]

                if idx_sample % 2 == 0:
                    curr_pair_diff = np.abs(np_scores[chosen_feature][idx_sample, predicted_label] -
                                            np_scores[chosen_feature][idx_sample + 1, predicted_label])
                    samples_html.append(f"<div class='pair' "
                                        f"style='background-color: rgba(255, 106, 0, {curr_pair_diff})'>")

                samples_html.append(f"<div class='pair-example' "
                                    f"title='P(&#375; = {predicted_label}) = {curr_label_score:.4f}'>"
                                    f"{'+' if idx_sample % 2 == 0 else '-'} "
                                    f"{samples[chosen_feature][idx_sample]}"
                                    f"</div>")

                if idx_sample % 2 == 1:
                    samples_html.append("</div>")

            step_html.append("<div id='step-{}-samples' style='display: none;'>{}</div>".format(idx_step, "\n".join(samples_html)))

        curr_diff = scores[chosen_feature][_cursor][predicted_label] - scores[chosen_feature][_cursor + 1][predicted_label]
        accounted_samples[chosen_feature] += 1
        samples_counter += 1

        updated_mean = importance_means[chosen_feature] + \
                       (curr_diff - importance_means[chosen_feature]) / accounted_samples[chosen_feature]
        updated_var = importance_vars[chosen_feature] + \
                      (curr_diff - importance_means[chosen_feature]) * (curr_diff - updated_mean)

        importance_means[chosen_feature] = updated_mean
        importance_vars[chosen_feature] = updated_var

        if idx_step % track_every_n_steps == 0:
            max_abs_imps = np.max(np.abs(importance_means))
            scaled_imps = scale_interval(importance_means, curr_low=-max_abs_imps, curr_high=max_abs_imps)

            for idx_tok, scaled_imp in enumerate(scaled_imps):
                curr_color = f"rgba(0, 153, 0, {scaled_imp})" if scaled_imp > 0 else f"rgba(255, 0, 0, {abs(scaled_imp)})"
                imp_sd = f"{np.sqrt(importance_vars[idx_tok] / accounted_samples[idx_tok]): .4f}" \
                    if accounted_samples[idx_tok] > 0 else "N/A"

                step_html.append(f"<span class='example-unit' "
                                 f"id='token-0-{idx_tok}'"
                                 f"title='{importance_means[idx_tok]: .4f} &#177; {imp_sd}' "
                                 f"style='background-color: {curr_color}'>"
                                 f"{sequence[idx_tok]}"
                                 f"</span>")

            example_html.append("<div id='step-{}'>{}</div>".format(idx_step, "\n".join(step_html)))
        idx_step += 1

    body_html = "<div class='example'>{}</div>".format("\n".join(example_html))
    return base_visualization(body_html, path=path)


if __name__ == "__main__":
    # highlight_plot(sequences=[["The", "show", "was", "fucking", "shite", "but", "I", "liked", "the", "end"]],
    #                importances=[[0.0, -0.05, 0.0, 0.35, 0.05, 0.01, 0.0, -0.1, 0.0, -0.05]],
    #                pred_labels=["toxic"], actual_labels=["toxic"], path="tmp_refactor.html")

    ime_run1 = MethodData.load("/home/matej/Downloads/SNLI_first_ex_comparison/snli_xs_ex0_accurate_ime/ime_data.json")
    track_progress(method_data=ime_run1,
                   idx_example=0,
                   path="tmp_refactor.html")

