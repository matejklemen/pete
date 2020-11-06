import json
import os
from enum import Enum
from typing import Union, Optional, Dict, List

NA_STR = "N/A"
DEFAULT_METHOD_DATA_FNAME = "method_data"


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
        "num_samples": [10, 10, 10, 10, 10]
    })

    dummy_method.save("/home/matej/Desktop", "joze")
    print(MethodData.load("/home/matej/Desktop/joze.json").serialize())
