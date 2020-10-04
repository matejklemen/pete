import torch
import logging


def estimate_feature_samples(contribution_variances: torch.Tensor, alpha: float, max_abs_error: float):
    normal = torch.distributions.Normal(0, 1)
    z_score = normal.icdf(torch.tensor(1.0 - alpha / 2))

    return ((z_score ** 2 * contribution_variances) / max_abs_error ** 2).int()


def estimate_max_samples(contribution_variances: torch.Tensor, alpha: float, max_abs_error: float):
    return torch.sum(estimate_feature_samples(contribution_variances, alpha, max_abs_error))


class IMEExplainer:
    def __init__(self, sample_data, model_func=None):
        self.model = model_func
        self.sample_data = sample_data
        self.num_features = self.sample_data.shape[1]

    def estimate_feature_importance(self, idx_feature: int, instance: torch.Tensor, num_samples: int):
        # Note: instance is currently supposed to be of shape [1, num_features]
        num_features = int(instance.shape[1])
        indices = torch.arange(num_features)

        if num_features != self.num_features:
            raise ValueError(f"Number of features in instance ({num_features}) "
                             f"does not match number of features in sampling data ({self.num_features})")

        samples = torch.zeros((2 * num_samples, num_features), dtype=instance.dtype)
        for idx_sample in range(num_samples):
            indices = indices[torch.randperm(num_features)]
            feature_pos = int(torch.nonzero(indices == idx_feature, as_tuple=False))
            idx_rand = int(torch.randint(self.sample_data.shape[0], size=()))

            # With feature `idx_feature` set
            samples[2 * idx_sample, :] = self.sample_data[idx_rand]
            samples[2 * idx_sample, indices[:feature_pos + 1]] = instance[0, indices[:feature_pos + 1]]

            # With feature `idx_feature` randomized
            samples[2 * idx_sample + 1, :] = self.sample_data[idx_rand]
            samples[2 * idx_sample + 1, indices[:feature_pos]] = instance[0, indices[:feature_pos]]

        scores = self.model(samples)
        scores_with = scores[::2]
        scores_without = scores[1::2]
        assert scores_with.shape[0] == scores_without.shape[0]
        diff = scores_with - scores_without

        return torch.mean(diff, dim=0), torch.var(diff, dim=0)

    def explain(self, instance: torch.Tensor, label: int = 0, **kwargs):
        """ Explain a prediction for instance.

        Args
        ----
        instance: torch.Tensor (shape: [1, num_features])
            Instance that is being explained.
        label: int (default: 0)
            Predicted label for instance. Leave at 0 if prediction is a regression score.
        kwargs:
            Optional parameters:
            [1] perturbable_mask (torch.Tensor): mask, specifying features that can be perturbed;
            [2] min_samples_per_feature (int): minimum samples to be taken for explaining each perturbable feature;
            [3] max_samples (int): maximum samples to be taken combined across all perturbable features;
            [4] model_func (function): function that returns classification/regression scores for instances - overrides
                                        the model_func provided when instantiating explainer.
        """
        num_features = int(instance.shape[1])
        importance_means = torch.zeros(num_features, dtype=torch.float32)
        importance_vars = torch.zeros(num_features, dtype=torch.float32)

        model_func = kwargs.get("model_func", None)
        if model_func is not None:
            self.model = model_func

        if self.model is None:
            raise ValueError("Model function must be specified either when instantiating explainer or "
                             "when calling explain() for specific instance")

        # If user doesn't specify a mask of perturbable features, assume every feature can be perturbed
        perturbable_mask = kwargs.get("perturbable_mask", torch.ones(num_features, dtype=torch.bool))
        perturbable_inds = torch.arange(num_features)[perturbable_mask]
        num_perturbable = perturbable_inds.shape[0]

        min_samples_per_feature = kwargs.get("min_samples_per_feature", 100)
        max_samples = kwargs.get("max_samples", num_perturbable * min_samples_per_feature)
        assert max_samples >= num_perturbable * min_samples_per_feature

        samples_per_feature = torch.zeros(num_features, dtype=torch.long)
        samples_per_feature[perturbable_mask] = min_samples_per_feature
        # Artificially assign 1 sample to features which won't be perturbed, just to ensure safe division
        # (no samples will actually be taken for non-perturbable inputs)
        samples_per_feature[torch.logical_not(perturbable_mask)] = 1

        taken_samples = num_perturbable * min_samples_per_feature  # cumulative sum

        # Initial pass: every feature will use at least `min_samples_per_feature` samples
        for idx_feature in perturbable_inds.tolist():
            curr_mean, curr_var = self.estimate_feature_importance(idx_feature, instance,
                                                                   num_samples=int(samples_per_feature[idx_feature]))
            importance_means[idx_feature] = curr_mean[label]
            importance_vars[idx_feature] = curr_var[label]

        while taken_samples < max_samples:
            var_diffs = (importance_vars / samples_per_feature) - (importance_vars / (samples_per_feature + 1))
            idx_feature = int(torch.argmax(var_diffs))

            curr_imp, _ = self.estimate_feature_importance(idx_feature, instance, num_samples=1)
            curr_imp = curr_imp[label]
            samples_per_feature[idx_feature] += 1
            taken_samples += 1

            # Incremental mean and variance calculation - http://datagenetics.com/blog/november22017/index.html
            updated_mean = importance_means[idx_feature] + \
                           (curr_imp - importance_means[idx_feature]) / samples_per_feature[idx_feature]
            updated_var = importance_vars[idx_feature] + \
                          (curr_imp - importance_means[idx_feature]) * (curr_imp - updated_mean)

            importance_means[idx_feature] = updated_mean
            importance_vars[idx_feature] = updated_var

        # Convert from variance of the differences (sigma^2) to variance of the importances (sigma^2 / m)
        importance_vars /= samples_per_feature
        samples_per_feature[torch.logical_not(perturbable_mask)] = 0

        return importance_means, importance_vars


if __name__ == "__main__":
    # dummy call example, only for debugging purpose
    def dummy_func(X):
        return torch.randn((X.shape[0], 2))

    explainer = IMEExplainer(model_func=dummy_func,
                             sample_data=torch.randint(10, size=(10, 3)))

    importances, _ = explainer.explain(torch.tensor([[1, 4, 0]]), min_samples_per_feature=10, max_samples=1_000)
    print(importances)
