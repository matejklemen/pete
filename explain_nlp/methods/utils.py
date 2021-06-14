import itertools
from typing import List, Optional, Mapping
from warnings import warn

import torch


def handle_custom_features(custom_features: Optional[List[List[int]]], perturbable_mask: torch.Tensor,
                           position_mapping: Optional[Mapping[int, List[int]]] = None):
    """ Validates and processes custom features out of which the explanation will be built. Processing includes
    mapping them to features in generator's representation, done through `position_mapping`.

    If you want to use primary features instead of custom ones, set `custom_features` to None.

    If generator's representation is the same as model's representation, set `position_mapping=None`.

    Args:
        custom_features:
            List of non-overlapping feature groups to be explained
        perturbable_mask:
            Mask of features that can be perturbed ("modified"), shape: [1, num_features]. Only perturbable features
            can be present in `custom_features`
        position_mapping:
            How the shifted positions of perturbable model features should be mapped. See `model_to_generator` for
            an example of what shifted positions are

    Returns:
        Tuple ((custom_features, custom_features_mapped), allocated positions in explanation vector))
    """
    num_features = perturbable_mask.shape[1]
    perturbable_indices = torch.arange(num_features)[perturbable_mask[0]].tolist()
    feat_to_shifted_position = {idx_pert: i for i, idx_pert in enumerate(perturbable_indices)}

    mapping = position_mapping
    # By default, custom features are mapped to themselves, e.g. [[1], [2, 3]] gets mapped to [[1], [2, 3]]
    if position_mapping is None:
        mapping = {position: [int(i)] for position, i in enumerate(perturbable_indices)}

    feature_groups = []
    if custom_features is None:
        for idx_feature in perturbable_indices:
            new_feature = mapping[feat_to_shifted_position[idx_feature]]
            feature_groups.append(new_feature)

        used_indices = perturbable_indices
    else:
        num_additional = len(custom_features)

        cover_count = torch.zeros(num_features)
        free_features = perturbable_mask.clone()

        for curr_group in custom_features:
            if not torch.all(perturbable_mask[0, curr_group]):
                raise ValueError(f"At least one of the features in group {curr_group} is not perturbable")
            if torch.any(cover_count[curr_group] > 0):
                raise ValueError(f"Custom features are not allowed to overlap (feature group {curr_group} overlaps "
                                 f"with some other group)")
            cover_count[curr_group] += 1
            free_features[0, curr_group] = False

            mapped_group = []
            for idx_feature in curr_group:
                mapped_group.extend(mapping[feat_to_shifted_position[idx_feature]])
            feature_groups.append(mapped_group)

        uncovered_features = torch.nonzero(free_features, as_tuple=False)
        if uncovered_features.shape[0] > 0:
            warn("Some perturbable features are uncovered by provided feature groups, so the feature groups will be "
                 "expanded with them. To avoid this, either set the features as unperturbable, or cover them with a "
                 "feature group")

        for _, idx_feature in uncovered_features:
            free_features[0, idx_feature] = False
            custom_features.append([int(idx_feature)])
            feature_groups.append(mapping[feat_to_shifted_position[int(idx_feature)]])
            num_additional += 1

        used_indices = list(range(num_features, num_features + num_additional))

    return (custom_features, feature_groups), used_indices


def estimate_feature_samples(contribution_variances: torch.Tensor, alpha: float, max_abs_error: float):
    normal = torch.distributions.Normal(0, 1)
    z_score = normal.icdf(torch.tensor(1.0 - alpha / 2))

    return ((z_score ** 2 * contribution_variances) / max_abs_error ** 2).int()


def sample_permutations(upper: int, indices: torch.Tensor, num_permutations: int):
    probas = torch.zeros((num_permutations, upper))
    probas[:, indices] = 1 / indices.shape[0]
    return torch.multinomial(probas, num_samples=indices.shape[0])


def incremental_mean(curr_mean, new_value, n: int):
    """ Perform an incremental mean update. `n` is the number of samples including `new_value`. """
    return curr_mean + (new_value - curr_mean) / n


def incremental_var(curr_mean, curr_var, new_mean, new_value, n: int):
    """ Perform an incremental variance update. `n` is the number of samples including `new_value`. """
    assert n >= 2
    return (curr_var * (n - 1) + (new_value - curr_mean) * (new_value - new_mean)) / n


def tensor_indexer(obj: torch.Tensor, indices) -> torch.Tensor:
    return obj[indices]


def list_indexer(obj: List[List], indices) -> torch.Tensor:
    return torch.tensor(list(itertools.chain(*[obj[_i] for _i in indices])), dtype=torch.long)


def extend_tensor(curr_tensor: torch.Tensor, num_new=1):
    # e.g. [1, 2, 3], num_new=2 -> [1, 0, 0, 2, 3], where 0s are new elements
    num_ex, num_features = curr_tensor.shape
    extended_tensor = torch.zeros((num_ex, num_features + num_new), dtype=curr_tensor.dtype)
    extended_tensor[:, 0] = curr_tensor[:, 0]
    extended_tensor[:, 1 + num_new:] = curr_tensor[:, 1:]
    return extended_tensor