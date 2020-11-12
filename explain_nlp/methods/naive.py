from itertools import permutations, product
import torch
import warnings


class ExactShapleyExplainer:
    def __init__(self, model_func, feature_values):
        # feature_values... list of indexable iterables ([i][j] ... j-th possible value that attribute i can take)
        self.model = model_func
        self.feature_values = feature_values
        self.num_features = len(feature_values)

        # The expected value of model over the whole set of feature values is expensive to compute and required in
        # many computations, therefore cache it
        self._model_expectation = None
        self._num_uniq_values = [len(curr_values) for curr_values in self.feature_values]
        warnings.warn(f"{type(self).__name__} is very compute- and memory-intensive (exponential) and mostly serves "
                      f"as a sanity check or for very small (educational) examples")

    def conditional_expectation(self, instance: torch.Tensor, fixed_features: torch.Tensor):
        """ Compute the expected model prediction if part of features are fixed and the other part is perturbed across
        its entire domain of values.

        Args
        ----
        instance: torch.Tensor (shape: [1, num_features])
            Instance from which values for fixed features are taken.
        fixed_features: torch.Tensor (shape: [num_features])
            Indices of features whose values are fixed to values from `instance`.

        Example
        -------
        Computation of E(f(X1, X2)| X2 = 0) over [0, 1] x [0, 1] can be computed as

        >>> def rand_bin_classifier(data):
        >>>     r = torch.rand((data.shape[0], 2))
        >>>     return r / r.sum(dim=1).unsqueeze(1)
        >>> explainer = ExactShapleyExplainer(model_func=rand_bin_classifier, feature_values=[[0, 1], [0, 1]])
        >>> # The value of instance at index 1 will remain fixed, while value at index 0 will be perturbed
        >>> explainer.conditional_expectation(torch.tensor([[1, 0]]), fixed_features=torch.tensor([1]))
        """
        free_features = list(set(range(self.num_features)) - set(fixed_features.tolist()))
        if len(free_features) == self.num_features and self._model_expectation is not None:
            return self._model_expectation

        num_examples = 0
        for curr_feature in free_features:
            num_examples = max(1, num_examples) * self._num_uniq_values[curr_feature]

        # All features known, no perturbations needed
        if num_examples == 0:
            return self.model(instance).squeeze(0)

        samples = torch.repeat_interleave(instance, repeats=num_examples, dim=0)
        free_feature_values = [self.feature_values[idx_feature] for idx_feature in free_features]
        value_combos = product(*free_feature_values) if len(free_features) > 0 else []
        for idx_sample, feature_values in enumerate(value_combos):
            samples[idx_sample, free_features] = torch.tensor(feature_values)

        expected_value = torch.mean(self.model(samples), dim=0)
        if len(free_features) == self.num_features:
            self._model_expectation = expected_value

        return expected_value

    def prediction_difference(self, instance, fixed_features):
        baseline = self.conditional_expectation(instance, torch.tensor([]))
        return self.conditional_expectation(instance, fixed_features) - baseline

    def estimate_feature_importance(self, idx_feature: int, instance: torch.Tensor):
        num_features = int(instance.shape[1])
        indices = list(range(num_features))

        diffs = []
        for curr_perm in permutations(indices):
            _curr_perm = torch.tensor(curr_perm)
            feature_pos = int(torch.nonzero(_curr_perm == idx_feature, as_tuple=False))

            diff_with = self.prediction_difference(instance, fixed_features=_curr_perm[: feature_pos + 1])
            diff_without = self.prediction_difference(instance, fixed_features=_curr_perm[: feature_pos])
            diffs.append(diff_with - diff_without)

        return torch.mean(torch.stack(diffs), dim=0)

    def explain(self, instance, label=0):
        num_features = int(instance.shape[1])
        importance_means = torch.zeros(num_features, dtype=torch.float32)

        for idx_feature in range(num_features):
            importance_means[idx_feature] = self.estimate_feature_importance(idx_feature, instance)[label]

        res = {
            "importance": importance_means
        }

        return res


if __name__ == "__main__":
    def ideal_or(data):
        probas = torch.zeros((data.shape[0], 2), dtype=torch.float32)
        results = torch.logical_or(data[:, [0]], data[:, [1]]).long().squeeze()
        probas[torch.arange(data.shape[0]), results] = 1.0
        return probas

    explainer = ExactShapleyExplainer(ideal_or, feature_values=[[0, 1], [0, 1]])
    explained_instance = torch.tensor([[1, 0]])
    print(f"Instance: {explained_instance}")
    # importances, variances = explainer.explain(explained_instance, ideal_or(explained_instance))
    explainer.explain(explained_instance, label=1)
