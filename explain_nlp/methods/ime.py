import itertools
from typing import Optional, Union, Tuple, List, Dict
from warnings import warn

import torch
from copy import deepcopy
import logging

from explain_nlp.modeling.modeling_base import InterpretableModel
from explain_nlp.methods.utils import sample_permutations, incremental_mean, incremental_var, \
    list_indexer, estimate_feature_samples, handle_custom_features


class IMEExplainer:
    def __init__(self, sample_data: torch.Tensor, model: InterpretableModel,
                 data_weights: Optional[torch.FloatTensor] = None,
                 confidence_interval: Optional[float] = None, max_abs_error: Optional[float] = None,
                 return_num_samples: Optional[bool] = False, return_samples: Optional[bool] = False,
                 return_scores: Optional[bool] = False, criterion: Optional[str] = "squared_error",
                 shared_vocabulary: Optional[bool] = False):
        """ Explain instances using IME.

        Args:
            sample_data:
                Data, used to create perturbations of instances. Must have same number of features as the instance.
            model:
                Model to be interpreted at instance level.
            data_weights:
                Weights used for sampling data, from which masking values are taken. Must be of same shape as `sample_data`.
            confidence_interval:
                Constraint on how accurately the computed importances should be computed: "approximately
                `confidence_interval` * 100% of importances should fall within `max_abs_error` of the true importance."
            max_abs_error:
                Constraint on how accurately the computed importances should be computed (see `confidence_interval`).
            return_num_samples:
                Return number of taken samples to estimate feature importances.
            return_samples:
                Return the created perturbations used to estimate feature importances.
            return_scores:
                Return the scores (e.g. probabilities) returned by model for perturbed samples.
            criterion:
                Which criterion to use when allocating the samples after initial variance estimation. Can be either
                "absolute_error" or "squared_error".
            shared_vocabulary:
                Whether model and generator use the same vocabulary: setting this to True can prevent some unnecessary
                breaking of words.
        """
        self.model = model
        self.sample_data = sample_data
        self.weights = data_weights
        self.num_examples = len(self.sample_data)
        self.num_features = len(self.sample_data[0])
        self.confidence_interval = confidence_interval
        self.max_abs_error = max_abs_error

        self.return_num_samples = return_num_samples
        self.return_samples = return_samples
        self.return_scores = return_scores

        self.error_constraint_given = self.confidence_interval is not None and self.max_abs_error is not None
        self.indexer = list_indexer
        self.shared_vocabulary = shared_vocabulary

        # TODO: move to functions
        if criterion == "absolute_error":
            self.criterion = \
                lambda importance_vars, taken_samples: torch.sqrt(importance_vars / taken_samples) - torch.sqrt(importance_vars / (taken_samples + 1))
        elif criterion == "squared_error":
            self.criterion = \
                lambda importance_vars, taken_samples: (importance_vars / taken_samples) - (importance_vars / (taken_samples + 1))
        else:
            raise ValueError(f"Unsupported criterion: '{criterion}'")

    @staticmethod
    def _handle_sample_constraints(num_features: int, num_additional: int, used_feature_indices: List[int],
                                   exact_samples_per_feature: Optional[torch.Tensor] = None,
                                   min_samples_per_feature: Optional[int] = None,
                                   max_samples: Optional[int] = None):
        """ Handles the different ways to obtain an IME explanation, either
        (1) providing exact number of samples per feature,
        (2) providing minimum number of samples per feature or
        (3) providing minimum number of samples per feature and maximum total samples.

        An assumption made here is that `used_feature_indices` all lie in [0, num_features + num_additional).

        Returns:
            Number of samples to take for each feature, total number of samples
        """
        assert exact_samples_per_feature is not None or min_samples_per_feature is not None, \
            "Either 'min_samples_per_feature' or 'exact_samples_per_feature' must be provided"

        samples_per_feature = torch.zeros(num_features + num_additional, dtype=torch.long)

        # Option 1: provide exact number of samples to take per feature
        if exact_samples_per_feature is not None:
            assert exact_samples_per_feature.shape[1] == (num_features + num_additional)

            if torch.any(torch.lt(exact_samples_per_feature[0, used_feature_indices], 2)):
                warn("Taking less than 2 samples to estimate a feature importance will result in the variance being"
                     "undefined (nan). To avoid this, use at least 2 samples for estimation")

            samples_per_feature[used_feature_indices] = exact_samples_per_feature[0, used_feature_indices]
            eff_max_samples = int(torch.sum(exact_samples_per_feature[0, used_feature_indices]))
        else:
            num_used = len(used_feature_indices)
            assert num_used > 0, "Trying to allocate samples to no features ('used_feature_indices' is empty)"

            # Option 2: provide min_samples_per_feature and max_samples
            if max_samples is not None:
                samples_per_feature[used_feature_indices] = min_samples_per_feature
                eff_max_samples = max_samples
            # Option 3: provide min_samples per feature, set max_samples automatically
            else:
                samples_per_feature[used_feature_indices] = min_samples_per_feature
                eff_max_samples = num_used * min_samples_per_feature

            assert min_samples_per_feature >= 2  # otherwise variance isn't defined
            assert eff_max_samples >= num_used * min_samples_per_feature

        return samples_per_feature, eff_max_samples

    def update_sample_data(self, new_data: torch.Tensor, data_weights: Optional[torch.FloatTensor] = None):
        self.sample_data = new_data
        self.weights = data_weights
        self.num_features = new_data.shape[1]

    @torch.no_grad()
    def estimate_feature_importance(self, idx_feature: int, instance: torch.Tensor,
                                    num_samples: int, perturbable_mask: torch.Tensor,
                                    feature_groups: Optional[Union[torch.Tensor, List[List[int]]]] = None,
                                    **modeling_kwargs):
        """ Estimate importance of a single feature or a group of features for `instance` using `num_samples` samples,
        where each sample corresponds to a pair of perturbations (one with estimated feature set and another
        with estimated feature randomized).

        Args:
            idx_feature:
                Feature whose importance is estimated. If `feature_groups` is provided, `idx_feature` points to the
                position of the estimated custom feature.
            instance:
                Explained instance, shape: [1, num_features].
            num_samples:
                Number of samples to take.
            perturbable_mask:
                Mask of features that can be perturbed ("modified"), shape: [1, num_features].
            feature_groups:
                Groups that define which features are to be taken as an atomic unit (are to be perturbed together).
                If not provided, groups of single perturbable features are used.
            **modeling_kwargs:
                Additional modeling data (e.g. attention masks,...)
        """

        num_features = int(len(instance[0]))
        if num_features != self.num_features:
            raise ValueError(f"Number of features in instance ({num_features}) "
                             f"does not match number of features in sampling data ({self.num_features})")

        if feature_groups is None:
            eff_feature_groups = torch.arange(num_features)[perturbable_mask[0]].tolist()
            idx_superfeature = eff_feature_groups.index(idx_feature)
            eff_feature_groups = [[_i] for _i in eff_feature_groups]
        else:
            eff_feature_groups = feature_groups
            idx_superfeature = idx_feature

        est_instance_features = eff_feature_groups[idx_feature]

        # Permuted POSITIONS of (super)features inside `eff_feature_groups`
        indices = sample_permutations(upper=len(eff_feature_groups),
                                      indices=torch.arange(len(eff_feature_groups)),
                                      num_permutations=num_samples)
        feature_pos = torch.nonzero(indices == idx_superfeature, as_tuple=False)

        # If weights are not provided, sample uniformly
        data_weights = torch.ones(self.sample_data.shape[0], dtype=torch.float32) \
            if self.weights is None else self.weights
        randomly_selected = torch.multinomial(data_weights, num_samples=num_samples, replacement=True)

        samples = instance.repeat((2 * num_samples, 1))
        for idx_sample in range(num_samples):
            curr_feature_pos = int(feature_pos[idx_sample, 1])
            idx_rand = int(randomly_selected[idx_sample])

            # Get indices of perturbed primary units (e.g. subwords)
            changed_features = self.indexer(eff_feature_groups, indices[idx_sample, curr_feature_pos + 1:])

            # With feature `idx_feature` set
            samples[2 * idx_sample, changed_features] = self.sample_data[idx_rand, changed_features]

            # With feature `idx_feature` randomized
            samples[2 * idx_sample + 1, changed_features] = self.sample_data[idx_rand, changed_features]
            samples[2 * idx_sample + 1, est_instance_features] = self.sample_data[idx_rand, est_instance_features]

        scores = self.model.score(samples, **modeling_kwargs)
        scores_with = scores[::2]
        scores_without = scores[1::2]
        assert scores_with.shape[0] == scores_without.shape[0]
        diff = scores_with - scores_without

        results = {
            "diff_mean": torch.mean(diff, dim=0),
            "diff_var": torch.var(diff, dim=0)
        }

        if self.return_samples:
            results["samples"] = samples.tolist()

        if self.return_scores:
            results["scores"] = scores.tolist()

        return results

    def model_to_generator(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor,
                           **modeling_kwargs):
        """ Maps the shifted[1] positions of perturbable model features to positions of perturbable generator
        features.

        For example, If the model input is `["[CLS]", "unbelieveable"]` and the generator input is
        `["<BOS>", "<BOS>", "un", "be", "lieve", "able"]`, the mapping would be `{0: [2, 3, 4, 5]}`.


        [1] Positions are shifted so that they start counting from the first PERTURBABLE feature instead of any feature.
        For example, "I" in ["[CLS]", "I", "am", ...] would have position 0 because [CLS] is unperturbable.
        """
        num_features = input_ids.shape[1]
        perturbable_indices = torch.arange(num_features)[perturbable_mask[0]]

        return {
            "generator_instance": {
                "input_ids": input_ids,
                "perturbable_mask": perturbable_mask,
                "aux_data": modeling_kwargs
            },
            "mapping": {position: [int(i)] for position, i in enumerate(perturbable_indices)}
        }

    def explain(self, instance: Union[torch.Tensor, List], label: Optional[int] = 0,
                perturbable_mask: Optional[torch.Tensor] = None,
                min_samples_per_feature: Optional[int] = 100, max_samples: Optional[int] = None,
                exact_samples_per_feature: Optional[torch.Tensor] = None,
                custom_features: Optional[List[List[int]]] = None,
                **modeling_kwargs) -> Dict:
        """ Explain a prediction for given instance.

        Args:
        ----
        instance:
            Instance that is being explained. Shape: [1, num_features].
        label: int
            Predicted label for instance.
        perturbable_mask:
            Mask, specifying features that can be perturbed. If not given, all features of instance are assumed to be
            perturbable. Shape: [1, num_features].
        min_samples_per_feature:
            Minimum samples to be taken for each perturbable feature to estimate variance of importance.
        exact_samples_per_feature:
            Specific number of samples to take per feature. Shape: [1, num_features + num_additional]
        max_samples:
            Maximum samples to be taken combined across all perturbable features. This gets overriden if
            `confidence_interval` and `max_abs_error` are also provided at instantiation.
        custom_features:
            List of non-overlapping feature groups to be explained. Used for obtaining explanations of bigger units,
            such as sentences.
        """
        num_features, num_additional = len(instance[0]), 0
        eff_perturbable_mask = perturbable_mask if perturbable_mask is not None \
            else torch.ones((1, num_features), dtype=torch.bool)

        # In IME and IME+LM where model and generator use same vocabulary, no conversion of data is needed
        if self.shared_vocabulary:
            conversion_data = IMEExplainer.model_to_generator(self, instance, eff_perturbable_mask, **modeling_kwargs)
        else:
            conversion_data = self.model_to_generator(instance, eff_perturbable_mask, **modeling_kwargs)
        generator_instance = conversion_data["generator_instance"]

        (new_custom_features, feature_groups), used_inds = handle_custom_features(
            custom_features=custom_features,
            perturbable_mask=eff_perturbable_mask,
            position_mapping=conversion_data["mapping"]
        )
        # If not all perturbable features are covered, new custom features can get automatically added
        if new_custom_features is not None:
            custom_features = new_custom_features

        feature_to_group_index = dict(zip(used_inds, range(len(used_inds))))

        importance_means = torch.zeros(num_features + num_additional, dtype=torch.float32)
        importance_vars = torch.zeros(num_features + num_additional, dtype=torch.float32)

        empty_metadata = {}
        if self.return_samples:
            empty_metadata["samples"] = []
        if self.return_scores:
            empty_metadata["scores"] = []
        feature_debug_data = [deepcopy(empty_metadata) for _ in range(num_features + num_additional)]

        samples_per_feature, eff_max_samples = IMEExplainer._handle_sample_constraints(
            num_features=num_features, num_additional=num_additional, used_feature_indices=used_inds,
            min_samples_per_feature=min_samples_per_feature, max_samples=max_samples,
            exact_samples_per_feature=exact_samples_per_feature
        )
        # Temporarily set number of samples to 1 for features that do not require any samples to avoid division by 0
        is_zero_samples = torch.eq(samples_per_feature, 0)
        samples_per_feature[is_zero_samples] = 1

        taken_samples = torch.sum(samples_per_feature[used_inds])  # cumulative sum

        # Initial pass: either estimate variance or take exact number samples as provided by user
        for idx_feature in used_inds:
            res = self.estimate_feature_importance(feature_to_group_index[idx_feature],
                                                   feature_groups=feature_groups,
                                                   instance=generator_instance["input_ids"],
                                                   num_samples=samples_per_feature[idx_feature],
                                                   perturbable_mask=generator_instance["perturbable_mask"],
                                                   **generator_instance["aux_data"])
            importance_means[idx_feature] = res["diff_mean"][label]
            importance_vars[idx_feature] = res["diff_var"][label]

            if self.return_samples:
                feature_debug_data[idx_feature]["samples"].append(res["samples"])

            if self.return_scores:
                feature_debug_data[idx_feature]["scores"].append(res["scores"])

        # Calculate required samples to satisfy constraint and call method recursively
        if self.error_constraint_given and exact_samples_per_feature is None:
            required_samples_per_feature = estimate_feature_samples(importance_vars,
                                                                    alpha=(1 - self.confidence_interval),
                                                                    max_abs_error=self.max_abs_error).long()

            # Either take min_samples_per_feature or number of required samples, whichever is greater
            need_additional = torch.gt(required_samples_per_feature, samples_per_feature)
            _required_samples_per_feature = required_samples_per_feature.clone()
            _required_samples_per_feature[used_inds] = min_samples_per_feature
            _required_samples_per_feature[need_additional] = required_samples_per_feature[need_additional]

            return self.explain(instance, label, perturbable_mask=eff_perturbable_mask,
                                min_samples_per_feature=min_samples_per_feature, max_samples=max_samples,
                                exact_samples_per_feature=_required_samples_per_feature.unsqueeze(0),
                                custom_features=custom_features,
                                **modeling_kwargs)

        while taken_samples < eff_max_samples:
            var_diffs = self.criterion(importance_vars, samples_per_feature)
            idx_feature = int(torch.argmax(var_diffs))

            res = self.estimate_feature_importance(feature_to_group_index[idx_feature],
                                                   feature_groups=feature_groups,
                                                   instance=instance,
                                                   num_samples=1,
                                                   perturbable_mask=eff_perturbable_mask,
                                                   **modeling_kwargs)
            curr_imp = res["diff_mean"][label]
            samples_per_feature[idx_feature] += 1
            taken_samples += 1

            if self.return_samples:
                feature_debug_data[idx_feature]["samples"].append(res["samples"])

            if self.return_scores:
                feature_debug_data[idx_feature]["scores"].append(res["scores"])

            # Incremental mean and variance calculation - http://datagenetics.com/blog/november22017/index.html
            updated_mean = incremental_mean(curr_mean=importance_means[idx_feature],
                                            new_value=curr_imp,
                                            n=samples_per_feature[idx_feature])
            updated_var = incremental_var(curr_mean=importance_means[idx_feature],
                                          curr_var=importance_vars[idx_feature],
                                          new_mean=updated_mean,
                                          new_value=curr_imp,
                                          n=samples_per_feature[idx_feature])

            importance_means[idx_feature] = updated_mean
            importance_vars[idx_feature] = updated_var

        # Convert from variance of the differences (sigma^2) to variance of the importances (sigma^2 / m)
        importance_vars /= samples_per_feature
        samples_per_feature[is_zero_samples] = 0

        results = {
            "importance": importance_means,
            "var": importance_vars,
            "taken_samples": eff_max_samples
        }

        if self.return_num_samples:
            results["num_samples"] = samples_per_feature

        if self.return_samples:
            results["samples"] = [list(itertools.chain(*feature_data["samples"])) if feature_data["samples"] else None
                                  for feature_data in feature_debug_data]

        if self.return_scores:
            results["scores"] = [list(itertools.chain(*feature_data["scores"])) if feature_data["scores"] else None
                                 for feature_data in feature_debug_data]

        if custom_features is not None:
            results["custom_features"] = custom_features

        return results

    def explain_text(self, text_data: Union[str, Tuple[str, ...]], label: Optional[int] = 0,
                     min_samples_per_feature: Optional[int] = 100, max_samples: Optional[int] = None,
                     exact_samples_per_feature: Optional[torch.Tensor] = None,
                     pretokenized_text_data: Optional[Union[List[str], Tuple[List[str], ...]]] = None,
                     custom_features: Optional[List[List[int]]] = None):
        # Convert instance being interpreted to representation of interpreted model
        is_split_into_units = pretokenized_text_data is not None
        model_instance = self.model.to_internal([pretokenized_text_data if is_split_into_units else text_data],
                                                is_split_into_units=is_split_into_units)

        res = self.explain(model_instance["input_ids"], label, perturbable_mask=model_instance["perturbable_mask"],
                           min_samples_per_feature=min_samples_per_feature, max_samples=max_samples,
                           exact_samples_per_feature=exact_samples_per_feature,
                           custom_features=custom_features,
                           **model_instance["aux_data"])
        res["input"] = self.model.from_internal(model_instance["input_ids"],
                                                take_as_single_sequence=True,
                                                skip_special_tokens=False,
                                                return_tokens=True)[0]

        return res


if __name__ == "__main__":
    from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    model = InterpretableBertForSequenceClassification(
        model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased",
        tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased",
        batch_size=2,
        max_seq_len=41,
        device="cpu"
    )

    sample_data = model.tokenizer.batch_encode_plus(
        [
            "My name is Iron Man",
            "That is unbelievable",
            "This is an example of a slightly longer sequence",
            "This is a very very very very very very very very very very very very long sequence"
        ],
        max_length=41, padding="max_length", return_tensors="pt"
    )["input_ids"]

    print("Running IME")
    explainer = IMEExplainer(model=model, sample_data=sample_data,
                             return_num_samples=True,
                             return_samples=True,
                             return_scores=True)

    ex = "The big brown fox jumps over the lazy dog."
    res = explainer.explain_text(ex, label=2, min_samples_per_feature=10)
    print(res["importance"])

