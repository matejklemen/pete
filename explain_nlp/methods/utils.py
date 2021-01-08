import itertools
from typing import List

import torch


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