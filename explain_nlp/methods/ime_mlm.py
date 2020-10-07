import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM


# in BERT pretraining, 15% of the tokens are masked - increasing this number decreases the available context and
# likely the generated sequences
MLM_MASK_PROPORTION = 0.15

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class IMEMaskedLMExplainer:
    def __init__(self, model_func=None, pretrained_name_or_path="bert-base-uncased"):
        self.model = model_func
        self.mlm_generator = BertForMaskedLM.from_pretrained(pretrained_name_or_path).to(DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_name_or_path)
        self.mlm_generator.eval()

    def estimate_feature_importance(self, idx_feature: int, instance: torch.Tensor, num_samples: int,
                                    perturbable_mask: torch.Tensor):
        # Note: instance is currently supposed to be of shape [1, num_features]
        num_features = int(instance.shape[1])
        indices = torch.arange(num_features)[perturbable_mask[0]]

        samples = torch.zeros((2 * num_samples, num_features), dtype=instance.dtype)
        for idx_sample in range(num_samples):
            indices = indices[torch.randperm(indices.shape[0])]
            feature_pos = int(torch.nonzero(indices == idx_feature, as_tuple=False))

            # TODO: do masked language modeling after this loop, in larger batches (to speed up things)
            perturbed_features = indices[feature_pos:]
            samples[2 * idx_sample: 2 * idx_sample + 2, :] = instance
            boundary = max(int(MLM_MASK_PROPORTION * perturbed_features.shape[0]), 1)

            # Do first step manually to ensure the current feature gets perturbed to a unique value
            curr_masked_indices = perturbed_features[: boundary]
            samples[2 * idx_sample: 2 * idx_sample + 2, curr_masked_indices] = self.tokenizer.mask_token_id
            with torch.no_grad():
                res = self.mlm_generator(samples[2 * idx_sample: 2 * idx_sample + 2].to(DEVICE),
                                         return_dict=True)
            logits = res["logits"].cpu()
            preds = torch.argmax(logits, dim=-1)
            # the two samples are perturbed in the same way to minimize potential effect of other features
            samples[2 * idx_sample: 2 * idx_sample + 2, curr_masked_indices] = preds[0, curr_masked_indices]

            # Ensure that the feature `idx_feature` gets perturbed by essentially doing a top-p sampling among
            # tokens that are not equal to current feature value
            curr_feature_logits = logits[0, idx_feature]
            curr_feature_logits[instance[0, idx_feature]] = -float("inf")
            probabilities = F.softmax(curr_feature_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)

            samples[2 * idx_sample, idx_feature] = instance[0, idx_feature]
            samples[2 * idx_sample + 1, idx_feature] = next_token

            num_chunks = (perturbed_features.shape[0] - boundary + boundary - 1) // boundary
            for i in range(1, 1 + num_chunks):
                curr_masked_indices = perturbed_features[i * boundary: (i + 1) * boundary]
                samples[2 * idx_sample: 2 * idx_sample + 2, curr_masked_indices] = self.tokenizer.mask_token_id
                with torch.no_grad():
                    res = self.mlm_generator(samples[2 * idx_sample: 2 * idx_sample + 2].to(DEVICE),
                                             return_dict=True)
                preds = torch.argmax(res["logits"].cpu(), dim=-1)
                samples[2 * idx_sample: 2 * idx_sample + 2, curr_masked_indices] = preds[0, curr_masked_indices]

        scores = self.model(samples)
        scores_with = scores[::2]
        scores_without = scores[1::2]
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
            [1] perturbable_mask (torch.Tensor (shape: [1, num_features])): mask, specifying features that can be perturbed;
            [2] min_samples_per_feature (int): minimum samples to be taken for explaining each perturbable feature;
            [3] max_samples (int): maximum samples to be taken combined across all perturbable features;
            [4] model_func (function): function that returns classification/regression scores for instances - overrides
                                        the model_func provided when instantiating explainer.
        """
        num_features = int(instance.shape[1])
        importance_means = torch.zeros(num_features, dtype=torch.float32)
        importance_vars = torch.zeros(num_features, dtype=torch.float32)

        # print("Original instance:")
        # print(self.tokenizer.decode(instance[0]))

        model_func = kwargs.get("model_func", None)
        if model_func is not None:
            self.model = model_func

        if self.model is None:
            raise ValueError("Model function must be specified either when instantiating explainer or "
                             "when calling explain() for specific instance")

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
            curr_mean, curr_var = self.estimate_feature_importance(idx_feature, instance,
                                                                   num_samples=int(samples_per_feature[idx_feature]),
                                                                   perturbable_mask=perturbable_mask)
            importance_means[idx_feature] = curr_mean[label]
            importance_vars[idx_feature] = curr_var[label]

        while taken_samples < max_samples:
            var_diffs = (importance_vars / samples_per_feature) - (importance_vars / (samples_per_feature + 1))
            idx_feature = int(torch.argmax(var_diffs))

            curr_imp, _ = self.estimate_feature_importance(idx_feature, instance,
                                                           num_samples=1,
                                                           perturbable_mask=perturbable_mask)
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
        samples_per_feature[torch.logical_not(perturbable_mask[0])] = 0

        return importance_means, importance_vars


if __name__ == "__main__":
    # dummy call example, only for debugging purpose
    def dummy_func(X):
        return torch.randn((X.shape[0], 2))

    explainer = IMEMaskedLMExplainer(model_func=dummy_func,
                                     pretrained_name_or_path="bert-base-uncased")

    importances, _ = explainer.explain(torch.tensor([[1, 4, 0]]),
                                       min_samples_per_feature=10, max_samples=1_000)
    print(importances)
