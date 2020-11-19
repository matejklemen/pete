import json
import os
from enum import Enum
from typing import Union, Optional, Dict, List

import matplotlib.pyplot as plt
import numpy as np

NA_STR = "N/A"
DEFAULT_METHOD_DATA_FNAME = "method_data"
DEFAULT_COLORS = [
    [0, 255, 0],        # green
    [255, 255, 0],      # yellow
    [255, 0, 0],        # red
    [255, 0, 255],      # magenta
    [0, 0, 255],        # blue
    [0, 255, 255],      # cyan
    # Colors that are between the above ones in color chart and are a bit ambiguous, but still pretty good
    [128, 255, 0],  # chartreuse
    [255, 128, 0],      # orange
    [255, 0, 128],      # rose
    [128, 0, 255],      # violet
    [0, 128, 255],      # azure
    [0, 255, 128],       # spring green
]
DEFAULT_COLORS = [[r / 255, g / 255, b / 255] for r, g, b in DEFAULT_COLORS]
DEFAULT_MARKERS = ["o", "v", "s", "<", "p", ">", "P", "X", "D", "d", "*", 11]


def mae(x1, x2):
    return np.mean(np.abs(np.subtract(x1, x2)))


class MethodType(Enum):
    IME = 1
    INDEPENDENT_IME_LM = 2
    INDEPENDENT_IME_MLM = 3
    DEPENDENT_IME_MLM = 4


class MethodData:
    def __init__(self, method_type: Union[str, MethodType],
                 model_description: Dict,
                 generator_description: Dict,
                 min_samples_per_feature: int,
                 possible_labels: Optional[List] = None,
                 used_data: Optional[Dict] = None,
                 confidence_interval: Optional[float] = None,
                 max_abs_error: Optional[float] = None,
                 **existing_data):
        # General method description
        self.method_type: MethodType = method_type if isinstance(method_type, MethodType) else MethodType[method_type]
        self.model_description = model_description
        self.generator_description = generator_description
        self.min_samples_per_feature = min_samples_per_feature
        self.possible_labels = possible_labels if possible_labels is not None else []
        self.used_data = used_data if used_data is not None else {}
        self.confidence_interval = confidence_interval
        self.max_abs_error = max_abs_error

        # Data for examples on which method is ran
        self.sequences: List[List[str]] = existing_data.get("sequences", [])
        self.pred_probas: List[List[float]] = existing_data.get("pred_probas", [])
        self.pred_labels: List = existing_data.get("pred_labels", [])
        self.actual_labels: List = existing_data.get("actual_labels", [])

        self.importances: List[List[float]] = existing_data.get("importances", [])
        self.variances: List[List[float]] = existing_data.get("variances", [])
        assert len(self.importances) == len(self.variances) == len(self.sequences) == len(self.pred_labels)

        self.num_samples = existing_data.get("num_samples", [])
        self.samples = existing_data.get("samples", [])
        self.model_scores = existing_data.get("model_scores", [])
        self.num_estimated_samples: List[Optional[int]] = existing_data.get("num_estimated_samples", [])
        self.times: List[Optional[int]] = existing_data.get("times", [])

    def add_example(self, sequence: List[str], label,
                    importances: List[float], variances: List[float],
                    num_samples: List[int],
                    probas: Optional[List[float]] = None,
                    actual_label: Optional = None,
                    samples: Optional[List[Union[str, List[str]]]] = None,
                    model_scores: Optional[List[List[float]]] = None,
                    num_estimated_samples: Optional[int] = None,
                    time_taken: Optional[float] = None):
        self.sequences.append(sequence)
        self.pred_probas.append(probas if probas is not None else [])
        self.pred_labels.append(label)
        self.actual_labels.append(actual_label)

        self.importances.append(importances)
        self.variances.append(variances)

        self.num_samples.append(num_samples)
        self.samples.append(samples if samples is not None else [])
        self.model_scores.append(model_scores if model_scores is not None else [])
        self.num_estimated_samples.append(num_estimated_samples)
        self.times.append(time_taken)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        return {
            "sequence": self.sequences[item],
            "pred_probas": self.pred_probas[item],
            "pred_label": self.pred_labels[item],
            "actual_label": self.actual_labels[item],
            "importances": self.importances[item],
            "variances": self.variances[item],
            "num_samples": self.num_samples[item],
            "samples": self.samples[item],
            "model_scores": self.model_scores[item],
            "num_estimated_samples": self.num_estimated_samples[item],
            "time_taken": self.times[item]
        }

    def serialize(self) -> Dict:
        """ Convert data to structure, which can be saved into a JSON. This mostly means copying data and
        replacing Python's `None`s with empty lists or `N/A` strings. """
        return {
            # General method description
            "method_type": self.method_type.name,
            "model_description": self.model_description,
            "generator_description": self.generator_description,
            "min_samples_per_feature": self.min_samples_per_feature,
            "possible_labels": self.possible_labels,
            "used_data": self.used_data,
            "confidence_interval": self.confidence_interval,
            "max_abs_error": self.max_abs_error,
            # Data for examples on which method is ran
            "sequences": self.sequences,
            "pred_probas": self.pred_probas,
            "pred_labels": self.pred_labels,
            "actual_labels": list(map(lambda label: label if label is not None else NA_STR, self.actual_labels)),
            "importances": self.importances,
            "variances": self.variances,
            "num_samples": list(map(lambda ns: ns if ns is not None else NA_STR, self.num_samples)),
            "samples": self.samples,
            "model_scores": self.model_scores,
            "num_estimated_samples": self.num_estimated_samples,
            "times": list(map(lambda t: t if t is not None else NA_STR, self.times))
        }

    @staticmethod
    def load(load_path):
        with open(load_path, "r") as f:
            file_content = json.load(f)
        return MethodData(**file_content)

    def save(self, save_dir, file_name=DEFAULT_METHOD_DATA_FNAME):
        _file_name = file_name[:-5] if file_name.endswith(".json") else file_name
        with open(os.path.join(save_dir, f"{_file_name}.json"), "w") as f:
            json.dump(self.serialize(), fp=f, indent=4)


class MethodGroup:
    # TODO: I need to change this to filter out special tokens: although current methods are not wrong, they
    #  underestimate MAE due to averaging over (too) many tokens
    MAX_XTICKS = 10
    INVALID_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}

    def __init__(self, data_paths: List[str], method_labels: List[str],
                 reference_path: Optional[str] = None, reference_label: Optional[str] = None,
                 additional_filtered_tokens: Optional[List[str]] = None):
        self.methods = [MethodData.load(curr_path) for curr_path in data_paths]
        self.method_labels = method_labels

        self.ref_method = None
        self.ref_label = None
        if reference_path is not None:
            self.ref_method = MethodData.load(reference_path)
            self.ref_label = reference_label

        self.ignored_tokens = set(self.INVALID_TOKENS)
        if additional_filtered_tokens is not None:
            self.ignored_tokens |= set(self.INVALID_TOKENS)

    def plot_required_samples_wins(self, included_methods: Optional[List[int]] = None, **kwargs):
        """ Plot the number of examples for which method `i` (spanned across rows) requires less estimated samples to
        satisfy error constraint than method `j` (spanned across columns). Note: ties are not counted anywhere.

        For example, value in box (0, 3)

        Args:
        ----
        included_methods:
            Indices of loaded methods to include in the plot.
        kwargs:
            Mostly matplotlib-related style keywords: `save_path` (str).
        """
        eff_included_methods = included_methods if included_methods is not None else list(range(len(self.methods)))
        if len(eff_included_methods) == 0:
            raise ValueError("Tried to create a plot without data")

        included_method_labels = [self.method_labels[idx_method] for idx_method in eff_included_methods]
        assert len(included_method_labels) == len(eff_included_methods)

        save_path = kwargs.get("save_path", None)

        gathered_data = []
        for idx_method in eff_included_methods:
            gathered_data.append(self.methods[idx_method].num_estimated_samples)

        # [i, j]... for how many examples method `i` (e.g. IME+MLM) requires LESS samples than method `j` (e.g. IME)
        num_wins = np.zeros((len(eff_included_methods), len(eff_included_methods)), dtype=np.int32)
        for idx_row in range(len(eff_included_methods)):
            for idx_col in range(len(eff_included_methods)):
                num_wins[idx_row, idx_col] = \
                    sum(ns1 < ns2 for ns1, ns2 in zip(gathered_data[idx_row], gathered_data[idx_col]))

        _, ax = plt.subplots()
        ax.matshow(num_wins, cmap=plt.cm.get_cmap("YlGn"))
        for (row, col), val in np.ndenumerate(num_wins):
            # Note: cols are x-axis, rows are y-axis!
            ax.text(col, row, str(val), va="center", ha="center")

        plt.title("Method#1 (row) requires less samples than Method#2 (col)")
        plt.xticks(np.arange(num_wins.shape[1]), included_method_labels)
        plt.yticks(np.arange(num_wins.shape[0]), included_method_labels)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        plt.clf()

    def plot_required_samples(self, included_methods: Optional[List[int]] = None, **kwargs):
        """ Plot the amount of estimated required samples (given error constraints) for included methods.

        Args:
        -----
        included_methods:
            Indices of loaded methods to include in the plot.
        kwargs:
            Mostly matplotlib-related style keywords: `colors` (list), `markers` (list), `linestyle` (str),
            `save_path` (str).
        """
        MAX_METHODS = 8  # chosen subjectively
        eff_included_methods = included_methods if included_methods is not None else list(range(len(self.methods)))
        if len(eff_included_methods) == 0:
            raise ValueError("Tried to create a plot without data")
        if len(eff_included_methods) > MAX_METHODS:
            raise ValueError("Tried plotting too many methods")

        method_colors = kwargs.get("colors", DEFAULT_COLORS[:len(eff_included_methods)])
        method_markers = kwargs.get("markers", DEFAULT_MARKERS[:len(eff_included_methods)])
        save_path = kwargs.get("save_path", None)

        included_method_labels = [self.method_labels[idx_method] for idx_method in eff_included_methods]
        assert len(included_method_labels) == len(eff_included_methods)

        gathered_data = []
        for idx_method in eff_included_methods:
            gathered_data.append(self.methods[idx_method].num_estimated_samples)

        # All methods should have same amount of data points
        num_points = len(gathered_data[0])
        assert all(len(md) == num_points for md in gathered_data)

        x_axis = np.arange(num_points)
        for i in range(len(eff_included_methods)):
            plt.plot(x_axis, gathered_data[i],
                     marker=method_markers[i], color=method_colors[i],
                     linestyle=kwargs.get("linestyle", "none"),
                     markeredgewidth=0.5, markeredgecolor="black")

        tick_step = num_points // self.MAX_XTICKS
        plt.xticks(np.arange(0, num_points + 1, tick_step))
        plt.xlabel("Example")
        plt.ylabel("Required samples")
        plt.legend(included_method_labels)

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        plt.clf()

    def plot_required_samples_avg_diff(self, included_methods: Optional[List[int]] = None, **kwargs):
        """ Plot the average difference between methods (+- standard deviation) in required samples to satisfy error
        constraints.

        Args:
        -----
        included_methods:
            Indices of loaded methods to include in the plot.
        kwargs:
            Mostly matplotlib-related style keywords: `save_path` (str).
        """
        eff_included_methods = included_methods if included_methods is not None else list(range(len(self.methods)))
        if len(eff_included_methods) == 0:
            raise ValueError("Tried to create a plot without data")

        included_method_labels = [self.method_labels[idx_method] for idx_method in eff_included_methods]
        assert len(included_method_labels) == len(eff_included_methods)

        save_path = kwargs.get("save_path", None)

        gathered_data = []
        for idx_method in eff_included_methods:
            gathered_data.append(self.methods[idx_method].num_estimated_samples)

        # [i, j]... how many samples on average method `i` (e.g. IME+MLM) requires w.r.t method `j` (e.g. IME)
        # e.g. -100 means that method `i` on average requires 100 less samples than method `j`
        samples_diff_mean = np.zeros((len(eff_included_methods), len(eff_included_methods)))
        samples_diff_sd = np.zeros((len(eff_included_methods), len(eff_included_methods)))

        for idx_row in range(len(eff_included_methods)):
            for idx_col in range(len(eff_included_methods)):
                diffs = [ns1 - ns2 for ns1, ns2 in zip(gathered_data[idx_row], gathered_data[idx_col])]
                samples_diff_mean[idx_row, idx_col] = np.mean(diffs)
                samples_diff_sd[idx_row, idx_col] = np.std(diffs)

        _, ax = plt.subplots()
        # Note: plotting -1 * samples so that green corresponds to a low number and yellow to a high
        ax.matshow(-samples_diff_mean, cmap=plt.cm.get_cmap("YlGn"))
        for (row, col), val in np.ndenumerate(samples_diff_mean):
            # Note: cols are x-axis, rows are y-axis!
            ax.text(col, row, f"{val:.2f} +- {samples_diff_sd[row, col]:.2f}", va="center", ha="center")

        plt.title("Method#1 (row) requires on average __ samples w.r.t. Method#2 (col)")
        plt.xticks(np.arange(samples_diff_mean.shape[1]), included_method_labels)
        plt.yticks(np.arange(samples_diff_mean.shape[0]), included_method_labels)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        plt.clf()

    def plot_example_mae(self, included_methods: Optional[List[int]] = None, **kwargs):
        """Plot the example-wise mean absolute error of calculated importances w.r.t reference, given at instantiation.

        Args:
        -----
        included_methods:
            Indices of loaded methods to include in the plot.
        kwargs:
            Mostly matplotlib-related style keywords: `colors` (list), `markers` (list), `linestyle` (str),
            `save_path` (str).
        """
        MAX_METHODS = 8  # chosen subjectively
        eff_included_methods = included_methods if included_methods is not None else list(range(len(self.methods)))
        if len(eff_included_methods) == 0:
            raise ValueError("Tried to create a plot without data")
        if len(eff_included_methods) > MAX_METHODS:
            raise ValueError("Tried plotting too many methods")

        method_colors = kwargs.get("colors", DEFAULT_COLORS[:len(eff_included_methods)])
        method_markers = kwargs.get("markers", DEFAULT_MARKERS[:len(eff_included_methods)])
        save_path = kwargs.get("save_path", None)

        included_method_labels = [self.method_labels[idx_method] for idx_method in eff_included_methods]
        assert len(included_method_labels) == len(eff_included_methods)

        ref_data = self.ref_method.importances
        gathered_data = []
        for idx_method in eff_included_methods:
            gathered_data.append(self.methods[idx_method].importances)

        # All methods should have same amount of data points
        num_points = len(ref_data)
        assert all(len(md) == num_points for md in gathered_data)

        # All methods should have same-length sequences, otherwise:
        # (1) we might be comparing importances of different tokens,
        # (2) mean could be computed over different lengths, giving us wrong results.
        # Only the first sequence pair is checked as a quick "proxy" instead of every pair
        assert all(len(md[0]) == len(ref_data[0]) for md in gathered_data)

        # All methods should have same tokens (i.e. aligned)
        # Only the first sequence is checked as a quick "proxy"
        for idx_method in eff_included_methods:
            ref_seq = self.ref_method.sequences[0]
            method_seq = self.methods[idx_method].sequences[0]
            assert all(t1 == t2 for t1, t2 in zip(ref_seq, method_seq))

        mean_abs_errors = []
        for i in range(len(eff_included_methods)):
            mean_abs_errors.append([mae(gathered_data[i][idx_seq], ref_data[idx_seq]) for idx_seq in range(num_points)])

        x_axis = np.arange(num_points)
        for i in range(len(eff_included_methods)):
            plt.plot(x_axis, mean_abs_errors[i],
                     marker=method_markers[i], color=method_colors[i],
                     linestyle=kwargs.get("linestyle", "none"),
                     markeredgewidth=0.5, markeredgecolor="black")

        plt.title(f"Mean absolute error w.r.t. '{self.ref_label}'")
        tick_step = num_points // self.MAX_XTICKS
        plt.xticks(np.arange(0, num_points + 1, tick_step))
        plt.xlabel("Example")
        plt.ylabel("Mean absolute error")
        plt.legend(included_method_labels)

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        plt.clf()

    def plot_global_mae(self, included_methods: Optional[List[int]] = None, **kwargs):
        """Plot the dataset-wise mean absolute error of calculated importances w.r.t reference, given at instantiation.
        I.e. the errors are taken over all tokens of all sequences and then averaged.

        Args:
        -----
        included_methods:
            Indices of loaded methods to include in the plot.
        kwargs:
            Mostly matplotlib-related style keywords: `save_path` (str).
        """
        eff_included_methods = included_methods if included_methods is not None else list(range(len(self.methods)))
        if len(eff_included_methods) == 0:
            raise ValueError("Tried to create a plot without data")

        save_path = kwargs.get("save_path", None)

        included_method_labels = [self.method_labels[idx_method] for idx_method in eff_included_methods]
        assert len(included_method_labels) == len(eff_included_methods)

        ref_data = self.ref_method.importances
        gathered_data = []
        for idx_method in eff_included_methods:
            gathered_data.append(self.methods[idx_method].importances)

        # All methods should have same amount of data points
        num_points = len(ref_data)
        assert all(len(md) == num_points for md in gathered_data)

        # All methods should have same-length sequences, otherwise:
        # (1) we might be comparing importances of different tokens,
        # (2) mean could be computed over different lengths, giving us wrong results.
        # Only the first sequence pair is checked as a quick "proxy" instead of every pair
        assert all(len(md[0]) == len(ref_data[0]) for md in gathered_data)

        # All methods should have same tokens (i.e. aligned)
        # Only the first sequence is checked as a quick "proxy"
        for idx_method in eff_included_methods:
            ref_seq = self.ref_method.sequences[0]
            method_seq = self.methods[idx_method].sequences[0]
            assert all(t1 == t2 for t1, t2 in zip(ref_seq, method_seq))

        glob_mean_abs_errors = []
        glob_sd_abs_errors = []
        for i in range(len(eff_included_methods)):
            curr_abs_errors = []
            for idx_seq in range(num_points):
                curr_abs_errors.extend(np.abs(np.subtract(gathered_data[i][idx_seq], ref_data[idx_seq])))

            glob_mean_abs_errors.append(np.mean(curr_abs_errors))
            glob_sd_abs_errors.append(np.std(curr_abs_errors))

        x_axis = np.arange(len(eff_included_methods))
        plt.title(f"Dataset-wise mean absolute error w.r.t. '{self.ref_label}'")
        plt.xticks(x_axis, included_method_labels)
        plt.ylabel("Mean absolute error")
        plt.bar(x_axis, glob_mean_abs_errors, yerr=glob_sd_abs_errors)

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        plt.clf()


if __name__ == "__main__":
    dummy_method = MethodData(
        method_type=MethodType.DEPENDENT_IME_MLM,
        model_description={},
        generator_description={},
        min_samples_per_feature=10,
        used_data=None,
        confidence_interval=0.99,
        max_abs_error=0.01
    )

    dummy_method.add_example(**{
        "sequence": ["My", "name", "is", "Iron", "Man"],
        "label": ["English"],
        "importances": [0.2, 0.3, 0.2, 0.4, 0.4],
        "variances": [0.0023, 0.1, 0.05, 0.03, 0.00006],
        "num_samples": [10, 10, 10, 10, 10],
        "num_estimated_samples": 100
    })
    dummy_method.add_example(**{
        "sequence": ["My", "name", "is", "Iron", "Man"],
        "label": ["English"],
        "importances": [0.2, 0.3, 0.2, 0.4, 0.4],
        "variances": [0.0023, 0.1, 0.05, 0.03, 0.00006],
        "num_samples": [10, 10, 10, 10, 10],
        "num_estimated_samples": 115
    })

    dummy_method.save("/home/matej/Desktop", "joze")
    print(MethodData.load("/home/matej/Desktop/joze.json").serialize())

