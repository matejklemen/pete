import torch
from copy import deepcopy

from explain_nlp.methods.utils import estimate_max_samples


class IMEExplainer:
    def __init__(self, sample_data, model_func=None,
                 return_variance=False, return_num_samples=False, return_samples=False, return_scores=False):
        self.model = model_func
        self.sample_data = sample_data
        self.num_features = self.sample_data.shape[1]

        self.return_variance = return_variance
        self.return_num_samples = return_num_samples
        self.return_samples = return_samples
        self.return_scores = return_scores

    def estimate_feature_importance(self, idx_feature: int, instance: torch.Tensor, num_samples: int,
                                    perturbable_mask: torch.Tensor):
        # Note: instance is currently supposed to be of shape [1, num_features]
        num_features = int(instance.shape[1])
        perturbable_inds = torch.arange(num_features)[perturbable_mask[0]]
        num_perturbable = int(perturbable_inds.shape[0])

        if num_features != self.num_features:
            raise ValueError(f"Number of features in instance ({num_features}) "
                             f"does not match number of features in sampling data ({self.num_features})")

        # Simulate batched permutation sampling
        probas = torch.zeros((num_samples, num_features))
        probas[:, perturbable_inds] = 1 / num_perturbable
        indices = torch.multinomial(probas, num_samples=num_perturbable)
        feature_pos = torch.nonzero(indices == idx_feature, as_tuple=False)

        samples = instance.repeat((2 * num_samples, 1))
        for idx_sample in range(num_samples):
            curr_feature_pos = int(feature_pos[idx_sample, 1])
            idx_rand = int(torch.randint(self.sample_data.shape[0], size=()))

            # With feature `idx_feature` set
            samples[2 * idx_sample, indices[idx_sample, curr_feature_pos + 1:]] = \
                self.sample_data[idx_rand, indices[idx_sample, curr_feature_pos + 1:]]

            # With feature `idx_feature` randomized
            samples[2 * idx_sample + 1, indices[idx_sample, curr_feature_pos:]] = \
                self.sample_data[idx_rand, indices[idx_sample, curr_feature_pos:]]

        scores = self.model(samples)
        scores_with = scores[::2]
        scores_without = scores[1::2]
        assert scores_with.shape[0] == scores_without.shape[0]
        diff = scores_with - scores_without

        results = {
            "diff_mean": torch.mean(diff, dim=0),
            "diff_var": torch.var(diff, dim=0)
        }

        if self.return_samples:
            results["samples"] = samples

        if self.return_scores:
            results["scores"] = scores

        return results

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
            [1] perturbable_mask (torch.Tensor (shape: [1, num_features])): mask, specifying features that can be perturbed;
            [2] min_samples_per_feature (int): minimum samples to be taken for explaining each perturbable feature;
            [3] max_samples (int): maximum samples to be taken combined across all perturbable features;
            [4] model_func (function): function that returns classification/regression scores for instances - overrides
                                        the model_func provided when instantiating explainer.
        """
        model_func = kwargs.get("model_func", None)
        if model_func is not None:
            self.model = model_func

        if self.model is None:
            raise ValueError("Model function must be specified either when instantiating explainer or "
                             "when calling explain() for specific instance")

        num_features = int(instance.shape[1])
        importance_means = torch.zeros(num_features, dtype=torch.float32)
        importance_vars = torch.zeros(num_features, dtype=torch.float32)

        empty_metadata = {}
        if self.return_samples:
            empty_metadata["samples"] = []
        if self.return_scores:
            empty_metadata["scores"] = []
        feature_debug_data = [deepcopy(empty_metadata) for _ in range(num_features)]

        # If user doesn't specify a mask of perturbable features, assume every feature can be perturbed
        perturbable_mask = kwargs.get("perturbable_mask", torch.ones((1, num_features), dtype=torch.bool))
        perturbable_inds = torch.arange(num_features)[perturbable_mask[0]]
        num_perturbable = perturbable_inds.shape[0]

        min_samples_per_feature = kwargs.get("min_samples_per_feature", 100)
        max_samples = kwargs.get("max_samples", num_perturbable * min_samples_per_feature)
        assert max_samples >= num_perturbable * min_samples_per_feature

        samples_per_feature = torch.zeros(num_features, dtype=torch.long)
        samples_per_feature[perturbable_mask[0]] = min_samples_per_feature
        # Artificially assign 1 sample to features which won't be perturbed, just to ensure safe division
        # (no samples will actually be taken for non-perturbable inputs)
        samples_per_feature[torch.logical_not(perturbable_mask[0])] = 1

        taken_samples = num_perturbable * min_samples_per_feature  # cumulative sum

        # Initial pass: every feature will use at least `min_samples_per_feature` samples
        for idx_feature in perturbable_inds.tolist():
            res = self.estimate_feature_importance(idx_feature, instance,
                                                   num_samples=int(samples_per_feature[idx_feature]),
                                                   perturbable_mask=perturbable_mask)
            importance_means[idx_feature] = res["diff_mean"][label]
            importance_vars[idx_feature] = res["diff_var"][label]

            if self.return_samples:
                feature_debug_data[idx_feature]["samples"].append(res["samples"])

            if self.return_scores:
                feature_debug_data[idx_feature]["scores"].append(res["scores"])

        confidence_interval = kwargs.get("confidence_interval", None)
        max_abs_error = kwargs.get("max_abs_error", None)
        constraint_given = confidence_interval is not None and max_abs_error is not None
        if constraint_given:
            max_samples = int(estimate_max_samples(importance_vars,
                                                   alpha=(1 - confidence_interval),
                                                   max_abs_error=max_abs_error))

        while taken_samples < max_samples:
            var_diffs = (importance_vars / samples_per_feature) - (importance_vars / (samples_per_feature + 1))
            idx_feature = int(torch.argmax(var_diffs))

            res = self.estimate_feature_importance(idx_feature, instance,
                                                   num_samples=1,
                                                   perturbable_mask=perturbable_mask)
            curr_imp = res["diff_mean"][label]
            samples_per_feature[idx_feature] += 1
            taken_samples += 1

            if self.return_samples:
                feature_debug_data[idx_feature]["samples"].append(res["samples"])

            if self.return_scores:
                feature_debug_data[idx_feature]["scores"].append(res["scores"])

            # Incremental mean and variance calculation - http://datagenetics.com/blog/november22017/index.html
            updated_mean = importance_means[idx_feature] + \
                           (curr_imp - importance_means[idx_feature]) / samples_per_feature[idx_feature]
            updated_var = importance_vars[idx_feature] + \
                          (curr_imp - importance_means[idx_feature]) * (curr_imp - updated_mean)

            importance_means[idx_feature] = updated_mean
            importance_vars[idx_feature] = updated_var

        # Convert from variance of the differences (sigma^2) to variance of the importances (sigma^2 / m)
        importance_vars /= samples_per_feature
        samples_per_feature[torch.logical_not(perturbable_mask[0])] = 0

        results = {
            "importance": importance_means,
            "taken_samples": max_samples
        }

        if self.return_variance:
            results["var"] = importance_vars

        if self.return_num_samples:
            results["num_samples"] = samples_per_feature

        if self.return_samples:
            results["samples"] = [torch.cat(feature_data["samples"]) if feature_data["samples"] else None
                                  for feature_data in feature_debug_data]

        if self.return_scores:
            results["scores"] = [torch.cat(feature_data["scores"]) if feature_data["scores"] else None
                                 for feature_data in feature_debug_data]

        return results


if __name__ == "__main__":
    # dummy call example, only for debugging purpose
    def dummy_func(X):
        return torch.randn((X.shape[0], 2))

    explainer = IMEExplainer(model_func=dummy_func,
                             sample_data=torch.randint(10, size=(10, 3)),
                             return_variance=True,
                             return_num_samples=True,
                             return_samples=True,
                             return_scores=True)

    res = explainer.explain(torch.tensor([[1, 4, 0]]), perturbable_mask=torch.tensor([[True, True, True]]),
                            min_samples_per_feature=10, max_samples=1_000, confidence_interval=0.95, max_abs_error=0.001)
    print(res["importance"])
