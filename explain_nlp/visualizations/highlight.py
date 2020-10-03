import os


def highlight_plot(sequences: list, labels: list, importances: list, path: str = None):
    # Obtain path to directory of this module with regard to caller path
    call_path = __file__.split(os.path.sep)[:-1]
    with open(os.path.join(os.path.sep.join(call_path), "highlight.css"), "r") as f:
        css_properties = f.read()
    with open(os.path.join(os.path.sep.join(call_path), "highlight.js"), "r") as f:
        js_code = f.read()

    if len(sequences) != len(importances):
        raise ValueError(f"Got an unequal amount of sequences and importances of sequence elements "
                         f"({len(sequences)} sequences != {len(importances)} importances)")

    if len(sequences) != len(labels):
        raise ValueError(f"Got an unequal amount of sequences and labels "
                         f"({len(sequences)} sequences != {len(labels)} labels)")

    viz_calls = []
    for i, (curr_seq, curr_label, curr_imps) in enumerate(zip(sequences, labels, importances)):
        if len(curr_seq) != len(curr_imps):
            raise ValueError(f"Example #{i}: importance not provided for each sequence element "
                             f"({len(curr_seq)} sequence elements != {len(curr_imps)} )")

        viz_calls.append(f"visualizeExample({curr_seq}, {curr_label}, {curr_imps});")
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
            <div class="slide-container">
                <strong>|importance| threshold (WIP): </strong> <br />
                <input type="range" min="0" max="1" step="0.01" value="0" class="slider">
            </div>
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
