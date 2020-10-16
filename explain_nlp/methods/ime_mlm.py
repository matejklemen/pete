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
    def __init__(self, model_func=None, pretrained_name_or_path="bert-base-uncased", batch_size=2, top_p=0.9,
                 return_variance=False, return_num_samples=False, return_samples=False, return_scores=False,
                 num_generated_samples=10):
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

        self.num_generated_samples = num_generated_samples
        self.sample_data = None

    def estimate_feature_importance(self, idx_feature: int, instance: torch.Tensor, num_samples: int,
                                    perturbable_mask: torch.Tensor, **kwargs):
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
            idx_rand = int(torch.randint(self.num_generated_samples, size=()))

            # With feature `idx_feature` set
            samples[2 * idx_sample, indices[idx_sample, curr_feature_pos + 1:]] = \
                self.sample_data[idx_rand, indices[idx_sample, curr_feature_pos + 1:]]

            # With feature `idx_feature` randomized
            samples[2 * idx_sample + 1, indices[idx_sample, curr_feature_pos:]] = \
                self.sample_data[idx_rand, indices[idx_sample, curr_feature_pos:]]

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

    def generate_samples(self, instance: torch.Tensor, perturbable_mask: torch.Tensor, **kwargs):
        num_features = int(instance.shape[1])
        perturbable_inds = torch.arange(num_features)[perturbable_mask[0]]
        num_perturbable = perturbable_inds.shape[0]
        masked_samples = instance.repeat((self.num_generated_samples, 1))

        # Mask and predict all tokens, one token at a time, in different order - slightly diverse greedy decoding
        probas = torch.zeros((self.num_generated_samples, num_features))
        probas[:, perturbable_inds] = 1 / num_perturbable
        indices = torch.multinomial(probas, num_samples=num_perturbable)

        token_type_ids = kwargs.get("token_type_ids")
        attention_mask = kwargs.get("attention_mask")

        for i in range(num_perturbable):
            curr_masked = indices[:, i]
            masked_samples[torch.arange(self.num_generated_samples), curr_masked] = self.tokenizer.mask_token_id

            aux_data = {
                "token_type_ids": token_type_ids.repeat((self.mlm_batch_size, 1)).to(DEVICE),
                "attention_mask": attention_mask.repeat((self.mlm_batch_size, 1)).to(DEVICE)
            }

            num_batches = (self.num_generated_samples + self.mlm_batch_size - 1) // self.mlm_batch_size
            for idx_batch in range(num_batches):
                s_batch, e_batch = idx_batch * self.mlm_batch_size, (idx_batch + 1) * self.mlm_batch_size
                curr_input_ids = masked_samples[s_batch: e_batch]
                curr_batch_size = curr_input_ids.shape[0]

                generator_data = {
                    "input_ids": curr_input_ids.to(DEVICE),
                    "token_type_ids": aux_data["token_type_ids"][: curr_batch_size],
                    "attention_mask": aux_data["attention_mask"][: curr_batch_size]
                }

                res = self.mlm_generator(**generator_data, return_dict=True)
                logits = res["logits"][torch.arange(curr_batch_size), curr_masked[s_batch: e_batch]]
                greedy_preds = torch.argmax(logits, dim=-1, keepdim=True)  # shape: [curr_batch_size, 1]

                masked_samples[torch.arange(s_batch, s_batch + curr_batch_size),
                               curr_masked[s_batch: e_batch]] = greedy_preds[:, 0].cpu()

        self.sample_data = masked_samples

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
        kwargs["perturbable_mask"] = perturbable_mask

        self.generate_samples(instance, **kwargs)

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
                                                   **kwargs)
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
                                                   **kwargs)
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
                                     return_num_samples=True,
                                     num_generated_samples=100,
                                     batch_size=8)
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    example = tokenizer.encode_plus("My name is Iron Man", "I am Iron Man", return_tensors="pt",
                                    return_special_tokens_mask=True, max_length=15, padding="max_length")
    res = explainer.explain(example["input_ids"],
                            perturbable_mask=torch.logical_not(example["special_tokens_mask"]),
                            min_samples_per_feature=10,
                            token_type_ids=example["token_type_ids"], attention_mask=example["attention_mask"])
    print(res["importance"])
