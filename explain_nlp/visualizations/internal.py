from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

LIME_VIZ_CMAP = matplotlib.colors.ListedColormap(['white', 'royalblue'])


def _argsort_bin(b):
    # https://stackoverflow.com/a/46953582
    b_view = np.ascontiguousarray(b).view(np.dtype((np.void, b.dtype.itemsize * b.shape[1])))
    return np.argsort(b_view.ravel())


def visualize_lime_internals(sequence_tokens: List[str], token_mask, probabilities,
                             width_per_sample: Optional[float] = 0.1,
                             file_path: Optional[str] = None,
                             allow_large_samples: Optional[bool] = False,
                             sort_key: Optional[str] = None,
                             **kwargs):
    """ Plot predicted probability as a function of feature indicators used in LIME. The plot is a combination of
    two subplots:
    (1) scatterplot displaying the probabilities for samples (top)
    (2) binary heatmap displaying whether a word is present in a certain sample or not (*)

    (*) Note that this is the behaviour in default LIME. The actual interpretation of values 0/1 in `token_mask` is up
    to the caller.

    Args:
        sequence_tokens:
            List of words in the sequence. Shape: [num_samples]
        token_mask:
            0/1 matrix that determines whether a token is present in sample. Shape: [num_samples, num_tokens]
        probabilities:
            Probabilities for samples provided via `token_mask`. Shape: [num_samples]
        width_per_sample:
            Width (in inches) that each sample is assigned. Total figure width is num_samples * width_per_sample.
        file_path:
            Where to save the figure. If not provided, displays the plot on screen.
        allow_large_samples:
            Disable safeguard that prevents the user from visualizing too big samples.
        sort_key:
            Plot samples in ascending sorted order, guided by this key. Possible choices are
            'token_mask' (sort by binary value of masks), 'probabilities' (sort by probabilities) or None (don't sort).
        **kwargs:
            Additional plotting arguments (`ylabel`).
    """
    MAX_INTENDED_SAMPLES = 1000
    _token_mask = token_mask if isinstance(token_mask, np.ndarray) else np.array(token_mask)
    _probabilities = probabilities if isinstance(probabilities, np.ndarray) else np.array(probabilities)

    _token_mask = _token_mask.astype(np.bool)
    num_samples = _token_mask.shape[0]
    if sort_key == "token_mask":
        sort_inds = _argsort_bin(_token_mask)
    elif sort_key == "probabilities":
        sort_inds = np.argsort(_probabilities)
    else:
        sort_inds = np.arange(num_samples)

    _token_mask = _token_mask[sort_inds].T  # [num_tokens, num_samples]
    _probabilities = _probabilities[sort_inds]
    num_tokens = len(sequence_tokens)

    if num_samples > MAX_INTENDED_SAMPLES and not allow_large_samples:
        raise ValueError(f"Visualization is not intended to be used with such a large sample "
                         f"({num_samples} > {MAX_INTENDED_SAMPLES}) as it can become extremely large and convoluted. "
                         f"To disable this check, pass `allow_large_samples=True`.")

    fig, ax = plt.subplots(2, 1, gridspec_kw={"wspace": 0.0, "hspace": 0.0})
    fig.set_figwidth(num_samples * width_per_sample)
    fig.subplots_adjust(wspace=0, hspace=0)

    ax[0].plot(np.arange(num_samples), _probabilities, "bo", linestyle="none")
    ax[0].set_xticks(np.arange(num_samples))
    ax[0].set_xticklabels([""] * num_samples)
    ax[0].set_yticks(np.arange(0.0, 1.0 + 1e-5, 0.2))
    ax[0].set_ylabel(kwargs.get("ylabel", "Probability"))
    ax[0].grid(which="both", linestyle="--")
    # Remove space before 0
    ax[0].margins(x=0)

    ax[1].pcolormesh(_token_mask.astype(np.int32)[::-1],
                     antialiased=True, cmap=LIME_VIZ_CMAP, edgecolors='black', linewidths=0.5)
    ax[1].set_xticks([])
    ax[1].set_yticks(np.arange(num_tokens) + 0.5)
    ax[1].set_yticklabels(sequence_tokens[::-1])
    ax[1].set_xlabel("Samples")

    plt.tight_layout()

    if file_path is not None:
        plt.savefig(file_path)
    else:
        plt.show()

    plt.clf()


if __name__ == "__main__":
    sequence = ["$This$", "$is$", "$the$", "$most$", "$useless$", "$product$", "$I$", "$have$", "$ever$", "$seen$"]
    mask = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    ])
    probas = np.array([
        0.9,
        0.2
    ])

    visualize_lime_internals(sequence, token_mask=mask, probabilities=probas,
                             width_per_sample=3.0, sort_key="token_mask")
