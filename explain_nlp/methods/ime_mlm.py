import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM
from copy import deepcopy


# in BERT pretraining, 15% of the tokens are masked - increasing this number decreases the available context and
# likely the quality of generated sequences
from explain_nlp.methods.utils import estimate_max_samples

MLM_MASK_PROPORTION = 0.15

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class IMEMaskedLMExplainer:
    def __init__(self, model_func=None, pretrained_name_or_path="bert-base-uncased", batch_size=16, top_p=0.9,
                 return_variance=False, return_num_samples=False, return_samples=False, return_scores=False):
        self.model = model_func
        self.mlm_batch_size = batch_size
        self.top_p = top_p

        self.return_variance = return_variance
        self.return_num_samples = return_num_samples
        self.return_samples = return_samples
        self.return_scores = return_scores

        self.mlm_generator = BertForMaskedLM.from_pretrained(pretrained_name_or_path).to(DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_name_or_path)
        self.mlm_generator.eval()

    def estimate_feature_importance(self, idx_feature: int, instance: torch.Tensor, num_samples: int,
                                    perturbable_mask: torch.Tensor):
        # Note: instance is currently supposed to be of shape [1, num_features]
        num_features = int(instance.shape[1])
        perturbable_inds = torch.arange(num_features)[perturbable_mask[0]]
        num_perturbable = int(perturbable_inds.shape[0])

        # Simulate batched permutation generation
        probas = torch.zeros((num_samples, num_features))
        probas[:, perturbable_inds] = 1 / num_perturbable
        indices = torch.multinomial(probas, num_samples=num_perturbable)
        feature_pos = torch.nonzero(indices == idx_feature, as_tuple=False)

        samples = instance.repeat((2 * num_samples, 1))
        for idx_sample in range(num_samples):
            curr_feature_pos = int(feature_pos[idx_sample, 1])

            # With feature `idx_feature` set
            samples[2 * idx_sample, indices[idx_sample, curr_feature_pos + 1:]] = self.tokenizer.mask_token_id

            # With feature `idx_feature` randomized
            samples[2 * idx_sample + 1, indices[idx_sample, curr_feature_pos:]] = self.tokenizer.mask_token_id

        # predicting `batch_size` examples at a time, but creating twice as many samples at a time
        # (they only differ in feature `idx_feature`)
        eff_batch_size = self.mlm_batch_size * 2
        max_masked_tokens = max(int(MLM_MASK_PROPORTION * num_features), 1)

        num_batches = (samples.shape[0] + eff_batch_size - 1) // eff_batch_size
        num_chunks = (num_perturbable + max_masked_tokens - 1) // max_masked_tokens

        for idx_batch in range(num_batches):
            s_batch, e_batch = idx_batch * eff_batch_size, (idx_batch + 1) * eff_batch_size
            curr_samples = samples[s_batch: e_batch]
            curr_batch = curr_samples[1::2].to(DEVICE)
            _local_batch = instance.repeat_interleave(curr_batch.shape[0], dim=0).to(DEVICE)

            for idx_chunk in range(num_chunks):
                s_chunk, e_chunk = idx_chunk * max_masked_tokens, (idx_chunk + 1) * max_masked_tokens
                curr_perturbed = perturbable_inds[s_chunk: e_chunk]
                min_perturbed, max_perturbed = curr_perturbed[0], curr_perturbed[-1]

                # Mask a chunk of the tokens
                _local_batch[:, curr_perturbed] = curr_batch[:, curr_perturbed]

                with torch.no_grad():
                    res = self.mlm_generator(_local_batch, return_dict=True)

                preds = torch.argmax(res["logits"], dim=-1)
                predict_mask = curr_batch == self.tokenizer.mask_token_id
                predict_mask[:, :min_perturbed] = False
                predict_mask[:, max_perturbed + 1:] = False

                _local_batch[predict_mask] = preds[predict_mask]

                # Increase chance that `idx_feature` gets perturbed to a different value by using top-p sampling
                if min_perturbed <= idx_feature <= max_perturbed:
                    curr_feature_logits = res["logits"][:, idx_feature]

                    if self.top_p:
                        sorted_logits, sorted_indices = torch.sort(curr_feature_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > self.top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        nonzero_rowcols = torch.nonzero(sorted_indices_to_remove, as_tuple=False)
                        rows = nonzero_rowcols[:, 0]
                        cols = nonzero_rowcols[:, 1]
                        tokens_to_remove = sorted_indices[rows, cols]
                        curr_feature_logits[rows, tokens_to_remove] = -float("inf")

                        # curr_feature_logits[:, instance[0, idx_feature]] = -float("inf")
                        probabilities = F.softmax(curr_feature_logits, dim=-1)
                        feature_predictions = torch.multinomial(probabilities, 1)
                    else:
                        feature_predictions = torch.argmax(curr_feature_logits, dim=-1, keepdim=True)

                    _local_batch[:, idx_feature] = feature_predictions[:, 0]

            _local_batch = _local_batch.cpu()
            # Examples are intertwined: one with fixed feature value and one without
            samples[s_batch: e_batch: 2] = _local_batch
            samples[s_batch: e_batch: 2, idx_feature] = instance[0, idx_feature]
            samples[s_batch + 1: e_batch: 2] = _local_batch

        scores = self.model(samples)
        scores_with = scores[::2]
        scores_without = scores[1::2]
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

    explainer = IMEMaskedLMExplainer(model_func=dummy_func,
                                     pretrained_name_or_path="bert-base-uncased",
                                     return_samples=True,
                                     return_scores=True,
                                     return_variance=True,
                                     return_num_samples=True)
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    example = tokenizer.encode_plus("My name is Iron Man", return_tensors="pt", return_special_tokens_mask=True)
    res = explainer.explain(example["input_ids"],
                            perturbable_mask=torch.logical_not(example["special_tokens_mask"]),
                            min_samples_per_feature=10, max_abs_error=0.01, confidence_interval=0.95)
    print(res["importance"])
