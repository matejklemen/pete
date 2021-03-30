import warnings
from typing import Union

import numpy as np
from sklearn.metrics import jaccard_score, precision_score


def fidelity(model_scores: Union[float, np.ndarray], surrogate_scores: Union[float, np.ndarray]):
    """ Measures how well a surrogate model approximates the original model.
    Assuming the scores are probabilities, fidelity ranges from 0.5 (worst) to 1.0 (best).

    Args:
        model_scores:
            Scores, predicted by the interpreted model
        surrogate_scores:
            Scores, predicted by the surrogate model
    """
    return 1.0 / (np.abs(model_scores - surrogate_scores) + 1.0)


def iou_score(y_true: np.ndarray, y_pred: np.ndarray):
    """ Compute intersection over union between ground truth sentence and predicted sentences, where each is a
    binary array.

    Args:
        y_true:
            One-hot array, where the nonzero element corresponds to the correct sentence
        y_pred:
            Binary array, whose elements are 1 if that sentence is predicted to be important
    """
    assert np.any(y_true)

    return jaccard_score(y_true, y_pred)


def hpd_score(sentence_ordering: np.ndarray, gt: np.ndarray):
    """ Compute highest precision for detection, which is the precision when all sentences with a score above
    (or equal to) ground truth sentence are taken as predictions by the method.

    Intuitively, this measures how many additional (false positive) sentences get predicted in addition to the correct
    sentence.

    Args:
        sentence_ordering:
            Array that specifies the ordering of sentence scores as provided by a method
        gt:
            One-hot array, where the nonzero element corresponds to the correct sentence
    """
    assert len(sentence_ordering) == len(gt)
    gt_bool = gt.astype(np.bool)

    # sentence_ranks[i]: in which place is the element `i` in `sentence_ordering`
    # e.g. sentence_ranks[3]: which rank does sentence 3 take?
    sentence_ranks = np.empty_like(sentence_ordering)
    sentence_ranks[sentence_ordering] = np.arange(len(sentence_ordering))

    gt_rank = sentence_ranks[gt_bool]
    if len(gt_rank) != 1:
        raise ValueError(f"Exactly one ground truth sentence is allowed "
                         f"(`gt` contains {len(gt_rank)} nonzero elements)")

    pred = np.zeros_like(gt_bool)
    pred[sentence_ordering[:int(gt_rank) + 1]] = True

    return precision_score(y_true=gt_bool, y_pred=pred)


def snr_score(sentence_scores: np.ndarray, gt: np.ndarray):
    """ Compute signal to noise ratio.

    Args:
        sentence_scores:
            Scores of sentences, assigned by a method
        gt:
            Binary array, in which 1 denotes a correct sentence and 0 a wrong sentence
    """
    gt_bool = gt.astype(np.bool)

    correct_score = sentence_scores[gt_bool]
    incorrect_scores = sentence_scores[np.logical_not(gt_bool)]

    if len(correct_score) != 1:
        raise ValueError(f"Exactly one ground truth sentence is allowed "
                         f"(`gt` contains {len(correct_score)} nonzero elements)")

    # If there is only one possible sentence, there are no incorrect scores which we need to take mean and sd of
    if len(incorrect_scores) == 0:
        incorrect_mean = 0.0
        incorrect_sd = 1.0
    else:
        incorrect_mean = np.mean(incorrect_scores)
        incorrect_sd = np.std(incorrect_scores)

    numerator = np.square(float(correct_score) - incorrect_mean)
    denominator = np.square(incorrect_sd)
    if denominator == 0.0:
        warnings.warn("Encountered zero deviation for incorrect sentence scores. "
                      "Setting denominator (sq. deviation) to 1.0 to avoid division by zero")
        denominator = 1.0

    return numerator / denominator
