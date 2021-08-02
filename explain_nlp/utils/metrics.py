import itertools
import warnings
from typing import Union, Dict, List

import numpy as np
import torch
from sklearn.metrics import jaccard_score, precision_score

from explain_nlp.modeling.modeling_base import InterpretableModel


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


def prediction_flip(interpreted_model: InterpretableModel, explanations: List[Dict],
                    texts=None, pretokenized_texts=None, allow_truncation=False):
    """ Computes the prediction flip metric. It measures the number of required deletions in order to go from the class
    predicted originally to some other class. The deletion candidates are explained units, taken in descending order
    of their importance.

    `explanations` is a list of dictionaries, containing "importance" (**required**, torch.Tensor) and
    "custom_features" (optional, List[List[int]]). It can be composed directly of outputs from explanation methods.

    **Either 'texts' or 'pretokenized_texts' must be provided**

    Args:
        interpreted_model:
            Model, which is being interpreted. Used to correctly transform text and score partially deleted input
        explanations:
            Explanations, obtained using some explanation method. Must contain key "importance" (torch.Tensor)
        texts:
            Raw text examples, to which `explanations` belongs
        pretokenized_texts:
            Pretokenized text examples, to which `explanations` belong. Useful when specific word boundaries need to be
            taken into account
        allow_truncation:
            Whether to allow truncating input sequence when converting to model's representation. Set to True if using
            texts that are not pre-truncated (i.e. obtained directly from explanations)

    Returns:
        dictionary, containing prediction flip metric based on number of deleted feature groups ("pred_flip_groups") or
         features ("pred_flip_feats")

    References:
        Nguyen, D. (2018). Comparing Automatic and Human Evaluation of Local Explanations for Text Classification.
        Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational
        Linguistics: Human Language Technologies, Volume 1 (Long Papers), 1069–1078.
        https://doi.org/10.18653/v1/N18-1097

    """
    use_pretok = pretokenized_texts is not None
    assert use_pretok or texts is not None, \
        "Either raw input text ('texts') or pretokenized input text ('pretokenized_texts') must be provided"

    used_input_text = pretokenized_texts if use_pretok else texts
    assert len(used_input_text) == len(explanations), \
        "Number of provided explanations must match number of input texts " \
        f"(got {len(explanations)} explanations and {len(used_input_text)} texts)"

    final_results = {
        "num_del_groups": [],
        "num_del_feats": [],
        "pred_flip_groups": [],
        "pred_flip_feats": []
    }

    for curr_example, curr_explanation in zip(used_input_text, explanations):
        used_feature_groups = curr_explanation.get("custom_features", None)
        encoded = interpreted_model.to_internal([curr_example],
                                                is_split_into_units=use_pretok,
                                                allow_truncation=allow_truncation)
        num_features = encoded["input_ids"].shape[1]
        perturbable_inds = torch.arange(num_features)[encoded["perturbable_mask"][0]]

        if used_feature_groups is None:
            used_feature_groups = [[_i.item()] for _i in perturbable_inds]
            used_importances = curr_explanation["importance"][perturbable_inds]
        else:
            used_importances = curr_explanation["importance"][num_features:]

        assert used_importances.shape[0] > 0, "There are no valid importances in the provided explanation"

        original_scores = interpreted_model.score(encoded["input_ids"], **encoded["aux_data"])
        original_pred = torch.argmax(original_scores[0])

        # elements where this is False get deleted
        keep_mask = torch.ones(num_features, dtype=torch.bool)
        num_del_groups, num_del_feats = 0, 0

        # Descending order: features most indicative of explained class are tried first
        sort_indices = torch.argsort(-used_importances).tolist()
        for rank, idx_group in enumerate(sort_indices, start=1):
            curr_group = used_feature_groups[idx_group]
            keep_mask[curr_group] = False
            num_del_groups += 1
            num_del_feats += len(curr_group)

            partial_input_ids = encoded["input_ids"][:, keep_mask]
            partial_aux_data = {k: v[:, keep_mask] for k, v in encoded["aux_data"].items()}

            new_scores = interpreted_model.score(partial_input_ids, **partial_aux_data)
            new_pred = torch.argmax(new_scores[0])

            if new_pred != original_pred:
                break

        final_results["num_del_groups"].append(num_del_groups)
        final_results["num_del_feats"].append(num_del_feats)
        final_results["pred_flip_groups"].append(num_del_groups / len(used_feature_groups))
        final_results["pred_flip_feats"].append(num_del_feats / perturbable_inds.shape[0])

    final_results["pred_flip_groups"] = sum(final_results["pred_flip_groups"]) / len(final_results["pred_flip_groups"])
    final_results["pred_flip_feats"] = sum(final_results["pred_flip_feats"]) / len(final_results["pred_flip_feats"])

    return final_results


def posthoc_accuracy(interpreted_model: InterpretableModel, explanations: List[Dict], top_k: Union[int, List[int]] = 1,
                     texts=None, pretokenized_texts=None, allow_truncation=False):
    """ Computes the post-hoc accuracy metric. It measures if the prediction using only top k most important units is
    the same as model's prediction using entire input. It is closely related to sufficiency.

    `explanations` is a list of dictionaries, containing "importance" (**required**, torch.Tensor) and
    "custom_features" (optional, List[List[int]]). It can be composed directly of outputs from explanation methods.

    **Either 'texts' or 'pretokenized_texts' must be provided**

    Args:
        interpreted_model:
            Model, which is being interpreted. Used to correctly transform text and score partially deleted input
        explanations:
            Explanations, obtained using some explanation method. Must contain key "importance" (torch.Tensor)
        top_k:
            Number of top units using which a prediction is made and compared to the original one. Can be a single
            number or multiple numbers, which are treated independently
        texts:
            Raw text examples, to which `explanations` belongs
        pretokenized_texts:
            Pretokenized text examples, to which `explanations` belong. Useful when specific word boundaries need to be
            taken into account
        allow_truncation:
            Whether to allow truncating input sequence when converting to model's representation. Set to True if using
            texts that are not pre-truncated (i.e. obtained directly from explanations)

    Returns:
        dictionary, containing the post-hoc accuracy for each k in `top_k`

    References:
        Chen, J., Song, L., Wainwright, M. J., & Jordan, M. I. (2018). Learning to Explain: An Information-Theoretic
        Perspective on Model Interpretation. ArXiv:1802.07814 [Cs, Stat]. http://arxiv.org/abs/1802.07814
    """
    use_pretok = pretokenized_texts is not None
    assert use_pretok or texts is not None, \
        "Either raw input text ('texts') or pretokenized input text ('pretokenized_texts') must be provided"

    used_input_text = pretokenized_texts if use_pretok else texts
    assert len(used_input_text) == len(explanations), \
        "Number of provided explanations must match number of input texts " \
        f"(got {len(explanations)} explanations and {len(used_input_text)} texts)"

    used_k_asc = [top_k] if isinstance(top_k, int) else sorted(top_k)
    final_results = {k: [] for k in used_k_asc}

    for curr_example, curr_explanation in zip(used_input_text, explanations):
        encoded = interpreted_model.to_internal([curr_example], is_split_into_units=use_pretok,
                                                allow_truncation=allow_truncation)

        original_scores = interpreted_model.score(encoded["input_ids"], **encoded["aux_data"])
        original_pred = torch.argmax(original_scores[0])

        num_features = encoded["input_ids"].shape[1]
        perturbable_inds = torch.arange(num_features)[encoded["perturbable_mask"][0]]

        used_feature_groups = curr_explanation.get("custom_features", None)
        if used_feature_groups is None:
            used_feature_groups = [[_i.item()] for _i in perturbable_inds]
            used_importances = curr_explanation["importance"][perturbable_inds]
        else:
            used_importances = curr_explanation["importance"][num_features:]

        # Element where this is True are kept
        # We always keep the unperturbable elements, as these include model conventions, such as prepending [CLS]
        keep_mask = torch.logical_not(encoded["perturbable_mask"][0])

        sort_indices = torch.argsort(-used_importances).tolist()
        for k in used_k_asc:
            kept_indices = sort_indices[:k]
            curr_features = list(itertools.chain(*[used_feature_groups[_g] for _g in kept_indices]))
            keep_mask[curr_features] = True

            partial_input_ids = encoded["input_ids"][:, keep_mask]
            partial_aux_data = {k: v[:, keep_mask] for k, v in encoded["aux_data"].items()}

            new_scores = interpreted_model.score(partial_input_ids, **partial_aux_data)
            new_pred = torch.argmax(new_scores[0])

            final_results[k].append(int(new_pred) == int(original_pred))

    for k in used_k_asc:
        final_results[k] = sum(final_results[k]) / len(final_results[k])

    return final_results
