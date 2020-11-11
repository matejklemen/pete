import os
from typing import List, Optional


def highlight_plot(sequences: List,
                   importances: List,
                   pred_labels: List,
                   actual_labels: Optional[List] = None,
                   path: Optional[str] = None):
    eff_actual_labels = [] if actual_labels is None else actual_labels
    if actual_labels is None:
        eff_actual_labels.append("")

    # Obtain path to directory of this module with regard to caller path
    call_path = __file__.split(os.path.sep)[:-1]
    with open(os.path.join(os.path.sep.join(call_path), "highlight.css"), "r") as f:
        css_properties = f.read()
    with open(os.path.join(os.path.sep.join(call_path), "highlight.js"), "r") as f:
        js_code = f.read()

    if len(sequences) != len(importances):
        raise ValueError(f"Got an unequal amount of sequences and importances of sequence elements "
                         f"({len(sequences)} sequences != {len(importances)} importances)")

    if len(sequences) != len(pred_labels):
        raise ValueError(f"Got an unequal amount of sequences and labels "
                         f"({len(sequences)} sequences != {len(pred_labels)} labels)")

    viz_calls = []
    for i, (curr_seq, curr_imps, curr_pred_label, curr_actual_label) in enumerate(zip(sequences, importances,
                                                                                      pred_labels, eff_actual_labels)):
        if len(curr_seq) != len(curr_imps):
            raise ValueError(f"Example #{i}: importance not provided for each sequence element "
                             f"({len(curr_seq)} sequence elements != {len(curr_imps)} )")

        viz_calls.append(f'visualizeExample({curr_seq}, "{curr_pred_label}", {curr_imps}, "{curr_actual_label}");')
    viz_calls = "\n".join(viz_calls)

    visualization = \
    f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style type="text/css">
        {css_properties}
        </style>
    </head>
    <body>
        <div class="examples-container">
            <div class="examples">
            </div>
        </div>
        <script type="text/javascript">
        {js_code}
        </script>
        <script type="text/javascript">
        {viz_calls}
        </script>
    </body>
    </html>
    """

    if path is not None:
        with open(path, "w") as f:
            print(f"Stored visualization into '{path}'")
            f.write(visualization)

    return visualization
