import torch


def estimate_feature_samples(contribution_variances: torch.Tensor, alpha: float, max_abs_error: float):
    normal = torch.distributions.Normal(0, 1)
    z_score = normal.icdf(torch.tensor(1.0 - alpha / 2))

    return ((z_score ** 2 * contribution_variances) / max_abs_error ** 2).int()


def estimate_max_samples(contribution_variances: torch.Tensor, alpha: float, max_abs_error: float):
    return torch.sum(estimate_feature_samples(contribution_variances, alpha, max_abs_error))

# TODO: sample_permutations()
