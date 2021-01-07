import itertools
from typing import Optional, Union, Tuple, List

import torch
from copy import deepcopy

from explain_nlp.methods.modeling import InterpretableModel
from explain_nlp.methods.utils import estimate_max_samples, sample_permutations, incremental_mean, incremental_var, \
    tensor_indexer, list_indexer, estimate_feature_samples


class IMEExplainer:
    def __init__(self, sample_data: torch.Tensor, model: InterpretableModel,
                 data_weights: Optional[torch.FloatTensor] = None,
                 confidence_interval: Optional[float] = None, max_abs_error: Optional[float] = None,
                 return_variance: Optional[bool] = False, return_num_samples: Optional[bool] = False,
                 return_samples: Optional[bool] = False, return_scores: Optional[bool] = False):
        """ Explain instances using IME.

        Args:
        -----
        sample_data: torch.Tensor
            Data, used to create perturbations of instances. Must have same number of features as the instance.
        model: InterpretableModel
            Model to be interpreted at instance level.
        data_weights: torch.Tensor (optional)
            Weights used for sampling data, from which masking values are taken. Must be of same shape as `sample_data`.
        confidence_interval: float (optional)
            Constraint on how accurately the computed importances should be computed: "approximately
            `confidence_interval` * 100% of importances should fall within `max_abs_error` of the true importance."
        max_abs_error: float
            Constraint on how accurately the computed importances should be computed (see `confidence_interval`).
        return_variance: bool
            Return variance of the importances.
        return_num_samples: bool
            Return number of taken samples to estimate feature importances.
        return_samples: bool
            Return the created perturbations used to estimate feature importances.
        return_scores: bool
            Return the scores (e.g. probabilities) returned by model for perturbed samples.
        """
        self.model = model
        self.sample_data = sample_data
        self.weights = data_weights
        self.num_examples = len(self.sample_data)
        self.num_features = len(self.sample_data[0])
        self.confidence_interval = confidence_interval
        self.max_abs_error = max_abs_error

        self.return_variance = return_variance
        self.return_num_samples = return_num_samples
        self.return_samples = return_samples
        self.return_scores = return_scores

        self.error_constraint_given = self.confidence_interval is not None and self.max_abs_error is not None
        self.indexer = tensor_indexer

    def update_sample_data(self, new_data: torch.Tensor, data_weights: Optional[torch.FloatTensor] = None):
        self.sample_data = new_data
        self.weights = data_weights
        self.num_features = new_data.shape[1]

    def estimate_feature_importance(self, idx_feature: Union[int, List[int]], instance: torch.Tensor, num_samples: int,
                                    perturbable_mask: torch.Tensor, label: Optional[str] = None,
                                    feature_groups: Optional[List[List[int]]] = None, **modeling_kwargs):
        # Note: instance is currently supposed to be of shape [1, num_features]
        num_features = int(len(instance[0]))

        # TODO: this is temporary
        # if isinstance(idx_feature, int):
        #     print(f"Estimating importance of '{self.model.tokenizer.decode([instance[0, idx_feature]])}'")
        # else:
        #     print(f"Estimating importance of '{self.model.tokenizer.decode(instance[0, idx_feature])}'")

        if num_features != self.num_features:
            raise ValueError(f"Number of features in instance ({num_features}) "
                             f"does not match number of features in sampling data ({self.num_features})")

        # Custom features present
        if feature_groups is not None:
            eff_feature_groups = feature_groups
            # Convert group of features to a new, "artificial" superfeature
            idx_superfeature = feature_groups.index(idx_feature)
        # Use regular, "primary" units (e.g. subwords)
        else:
            eff_feature_groups = torch.arange(num_features)[perturbable_mask[0]]
            idx_superfeature = eff_feature_groups.tolist().index(idx_feature)

        # Permuted POSITIONS of (super)features inside `eff_feature_groups`
        indices = sample_permutations(upper=len(eff_feature_groups),
                                      indices=torch.arange(len(eff_feature_groups)),
                                      num_permutations=num_samples)
        feature_pos = torch.nonzero(indices == idx_superfeature, as_tuple=False)

        # If weights are not provided, sample uniformly
        data_weights = torch.ones(self.sample_data.shape[0], dtype=torch.float32)
        if self.weights is not None:
            data_weights = self.weights[:, idx_feature]
            if data_weights.dim() > 1:
                data_weights = torch.mean(data_weights, dim=1)
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
            samples[2 * idx_sample + 1, idx_feature] = self.sample_data[idx_rand, idx_feature]

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

    def explain(self, instance: Union[torch.Tensor, List], label: Optional[int] = 0, perturbable_mask: Optional[torch.Tensor] = None,
                min_samples_per_feature: Optional[int] = 100, max_samples: Optional[int] = None,
                custom_features: Optional[List[List[int]]] = None,
                **modeling_kwargs):
        """ Explain a prediction for given instance.

        Args:
        ----
        instance:
            Instance that is being explained. Shape: [1, num_features].
        label: int
            Predicted label for instance. Leave at 0 if prediction is a regression score.
        perturbable_mask:
            Mask, specifying features that can be perturbed. If not given, all features of instance are assumed to be
            perturbable. Shape: [1, num_features].
        min_samples_per_feature:
            Minimum samples to be taken for each perturbable feature to estimate variance of importance.
        max_samples:
            Maximum samples to be taken combined across all perturbable features. This gets overriden if
            `confidence_interval` and `max_abs_error` are also provided at instantiation.
        custom_features:
            List of non-overlapping feature groups to be explained. Used for obtaining explanations of bigger units,
            such as sentences.
        """
        num_features = len(instance[0])
        num_additional = 0
        eff_perturbable_mask = perturbable_mask if perturbable_mask is not None \
            else torch.ones((1, num_features), dtype=torch.bool)
        self.indexer = tensor_indexer

        feature_groups = None
        superfeatures = list(range(num_features))
        if custom_features is not None:
            self.indexer = list_indexer
            num_additional = len(custom_features)

            cover_count = torch.zeros(num_features)
            for curr_group in custom_features:
                superfeatures.append(curr_group)
                if not torch.all(eff_perturbable_mask[0, curr_group]):
                    raise ValueError(f"At least one of the features in group {curr_group} is not perturbable")
                if torch.any(cover_count[curr_group] > 0):
                    raise ValueError(f"Custom features are not allowed to overlap (feature group {curr_group} overlaps "
                                     f"with some other group)")
                cover_count[curr_group] += 1

            feature_groups = []  # type: Optional[List[List[int]]]
            free_features = eff_perturbable_mask.clone()
            for curr_group in custom_features:
                free_features[0, curr_group] = False
                feature_groups.append(curr_group)

            for _, idx_feature in torch.nonzero(free_features, as_tuple=False):
                free_features[0, idx_feature] = False
                feature_groups.append([idx_feature.item()])

        importance_means = torch.zeros(num_features + num_additional, dtype=torch.float32)
        importance_vars = torch.zeros(num_features + num_additional, dtype=torch.float32)

        empty_metadata = {}
        if self.return_samples:
            empty_metadata["samples"] = []
        if self.return_scores:
            empty_metadata["scores"] = []
        feature_debug_data = [deepcopy(empty_metadata) for _ in range(num_features + num_additional)]

        perturbable_inds = torch.arange(num_features)[eff_perturbable_mask[0]].tolist() \
            if custom_features is None else list(range(num_features, num_features + num_additional))
        num_perturbable = len(perturbable_inds)

        eff_max_samples = max_samples if max_samples is not None else (num_perturbable * min_samples_per_feature)
        assert min_samples_per_feature >= 2  # otherwise variance isn't defined
        assert eff_max_samples >= num_perturbable * min_samples_per_feature

        samples_per_feature = torch.ones(num_features + num_additional, dtype=torch.long)
        samples_per_feature[perturbable_inds] = min_samples_per_feature
        samples_per_feature[num_features:] = min_samples_per_feature

        taken_samples = num_perturbable * min_samples_per_feature  # cumulative sum

        # Initial pass: every feature will use at least `min_samples_per_feature` samples
        for idx_feature in perturbable_inds:
            res = self.estimate_feature_importance(superfeatures[idx_feature],
                                                   instance=instance, label=label,
                                                   num_samples=samples_per_feature[idx_feature],
                                                   perturbable_mask=eff_perturbable_mask,
                                                   feature_groups=feature_groups,
                                                   **modeling_kwargs)
            importance_means[idx_feature] = res["diff_mean"][label]
            importance_vars[idx_feature] = res["diff_var"][label]

            if self.return_samples:
                feature_debug_data[idx_feature]["samples"].append(res["samples"])

            if self.return_scores:
                feature_debug_data[idx_feature]["scores"].append(res["scores"])

        if self.error_constraint_given:
            # Calculate required samples to satisfy constraint, making sure that if "too many" samples were already
            # taken for some feature (min_samples_per_feature > required_samples_per_feature[i]), that does not count
            # towards lowering the total amount of required samples
            required_samples_per_feature = estimate_feature_samples(importance_vars,
                                                                    alpha=(1 - self.confidence_interval),
                                                                    max_abs_error=self.max_abs_error)
            required_samples_per_feature -= samples_per_feature
            eff_max_samples = int(taken_samples + torch.sum(required_samples_per_feature[required_samples_per_feature > 0]))

        while taken_samples < eff_max_samples:
            var_diffs = (importance_vars / samples_per_feature) - (importance_vars / (samples_per_feature + 1))
            idx_feature = int(torch.argmax(var_diffs))

            res = self.estimate_feature_importance(superfeatures[idx_feature],
                                                   instance=instance, label=label,
                                                   num_samples=1,
                                                   perturbable_mask=eff_perturbable_mask,
                                                   feature_groups=feature_groups,
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
        _samples_per_feature = torch.zeros_like(samples_per_feature)
        _samples_per_feature[perturbable_inds] = samples_per_feature[perturbable_inds]

        results = {
            "importance": importance_means,
            "taken_samples": eff_max_samples
        }

        if self.return_variance:
            results["var"] = importance_vars

        if self.return_num_samples:
            results["num_samples"] = _samples_per_feature

        if self.return_samples:
            results["samples"] = [list(itertools.chain(*feature_data["samples"])) if feature_data["samples"] else None
                                  for feature_data in feature_debug_data]

        if self.return_scores:
            results["scores"] = [list(itertools.chain(*feature_data["scores"])) if feature_data["scores"] else None
                                 for feature_data in feature_debug_data]

        if custom_features is not None:
            results["custom_features"] = superfeatures[num_features:]

        return results

    def explain_text(self, text_data: Union[str, Tuple[str, ...]], label: Optional[int] = 0,
                     min_samples_per_feature: Optional[int] = 100, max_samples: Optional[int] = None,
                     pretokenized_text_data: Optional[Union[List[str], Tuple[List[str], ...]]] = None,
                     custom_features: Optional[List[List[int]]] = None):
        # Convert instance being interpreted to representation of interpreted model
        model_instance = self.model.to_internal([text_data],
                                                pretokenized_text_data=[pretokenized_text_data] if pretokenized_text_data is not None else None)

        res = self.explain(model_instance["input_ids"], label, perturbable_mask=model_instance["perturbable_mask"],
                           min_samples_per_feature=min_samples_per_feature, max_samples=max_samples,
                           custom_features=custom_features,
                           **model_instance["aux_data"])
        res["input"] = self.model.convert_ids_to_tokens(model_instance["input_ids"])[0]

        return res


class SequentialIMEExplainer(IMEExplainer):
    def __init__(self, sample_data: torch.Tensor, model: InterpretableModel,
                 data_weights: Optional[torch.FloatTensor] = None,
                 confidence_interval: Optional[float] = None, max_abs_error: Optional[float] = None,
                 return_variance: Optional[bool] = False, return_num_samples: Optional[bool] = False,
                 return_samples: Optional[bool] = False, return_scores: Optional[bool] = False):
        super().__init__(sample_data=sample_data, model=model, data_weights=data_weights,
                         confidence_interval=confidence_interval, max_abs_error=max_abs_error,
                         return_variance=return_variance, return_num_samples=return_num_samples,
                         return_samples=return_samples, return_scores=return_scores)

        self.special_token_ids = set(model.special_token_ids)

        # For each sequence in sampling data, mark down the indices where valid (non-special) tokens are present
        self.valid_indices = []
        _all_token_indices = list(range(self.sample_data.shape[1]))
        for idx_seq in range(self.sample_data.shape[0]):
            curr_valid = torch.tensor(list(filter(lambda i: self.sample_data[idx_seq, i].item() not in self.special_token_ids,
                                                  _all_token_indices)))
            if len(curr_valid) == 0:
                raise ValueError(f"Encountered sequence with no non-special token IDs (sequence#{idx_seq})")
            self.valid_indices.append(curr_valid)

    def estimate_feature_importance(self, idx_feature: Union[int, List[int]], instance: torch.Tensor, num_samples: int,
                                    perturbable_mask: torch.Tensor, label: Optional[str] = None,
                                    feature_groups: Optional[List[List[int]]] = None, **modeling_kwargs):
        # Note: instance is currently supposed to be of shape [1, num_features]
        num_features = int(len(instance[0]))

        # TODO: this is temporary
        # if isinstance(idx_feature, int):
        #     print(f"Estimating importance of '{self.model.tokenizer.decode([instance[0, idx_feature]])}'")
        # else:
        #     print(f"Estimating importance of '{self.model.tokenizer.decode(instance[0, idx_feature])}'")

        if num_features != self.num_features:
            raise ValueError(f"Number of features in instance ({num_features}) "
                             f"does not match number of features in sampling data ({self.num_features})")

        # Custom features present
        if feature_groups is not None:
            eff_feature_groups = feature_groups
            # Convert group of features to a new, "artificial" superfeature
            idx_superfeature = feature_groups.index(idx_feature)
        # Use regular, "primary" units (e.g. subwords)
        else:
            eff_feature_groups = torch.arange(num_features)[perturbable_mask[0]]
            idx_superfeature = eff_feature_groups.tolist().index(idx_feature)

        perturbable_inds = torch.arange(num_features)[perturbable_mask[0]]
        curr_min, curr_max = perturbable_inds[0], perturbable_inds[-1]
        # Permuted POSITIONS of (super)features inside `eff_feature_groups`
        indices = sample_permutations(upper=len(eff_feature_groups),
                                      indices=torch.arange(len(eff_feature_groups)),
                                      num_permutations=num_samples)
        feature_pos = torch.nonzero(indices == idx_superfeature, as_tuple=False)

        # If weights are not provided, sample uniformly
        data_weights = torch.ones(self.sample_data.shape[0], dtype=torch.float32)
        if self.weights is not None:
            data_weights = self.weights[:, idx_feature]
        randomly_selected = torch.multinomial(data_weights, num_samples=num_samples, replacement=True)

        samples = instance.repeat((2 * num_samples, 1))
        for idx_sample in range(num_samples):
            curr_feature_pos = int(feature_pos[idx_sample, 1])
            idx_rand = int(randomly_selected[idx_sample])
            new_max = len(self.valid_indices[idx_rand]) - 1

            # With feature `idx_feature` set
            indices_with = self.indexer(eff_feature_groups, indices[idx_sample, curr_feature_pos + 1:])
            mapped_indices_with = torch.floor_divide((indices_with - curr_min) * new_max,
                                                     curr_max - curr_min)
            mapped_indices_with = self.valid_indices[idx_rand][mapped_indices_with]
            samples[2 * idx_sample, indices_with] = self.sample_data[idx_rand, mapped_indices_with]

            # With feature `idx_feature` randomized
            indices_without = self.indexer(eff_feature_groups, indices[idx_sample, curr_feature_pos:])
            mapped_indices_without = torch.floor_divide((indices_without - curr_min) * new_max,
                                                        curr_max - curr_min)
            mapped_indices_without = self.valid_indices[idx_rand][mapped_indices_without]
            samples[2 * idx_sample + 1, indices_without] = self.sample_data[idx_rand, mapped_indices_without]

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
            results["samples"] = samples

        if self.return_scores:
            results["scores"] = scores

        return results


class WholeWordIMEExplainer(IMEExplainer):
    """ TODO: currently only intended for use with InterpretableBertForSequenceClassification (not generalized) """
    def estimate_feature_importance(self, idx_feature: int, instance: torch.Tensor, num_samples: int,
                                    perturbable_mask: torch.Tensor, label: Optional[str] = None,
                                    feature_groups: Optional[List[List[int]]] = None, **modeling_kwargs):
        if feature_groups is not None:  # TODO
            raise NotImplementedError("'feature_groups' is not supported in WholeWordIME and probably won't be, but "
                                      "the method needs to be verified that it's working properly.")

        num_features = len(instance[0])
        perturbable_inds = torch.arange(num_features)[perturbable_mask[0]]

        if num_features != self.num_features:
            raise ValueError(f"Number of features in instance ({num_features}) "
                             f"does not match number of features in sampling data ({self.num_features})")

        indices = sample_permutations(upper=num_features, indices=perturbable_inds,
                                      num_permutations=num_samples)
        feature_pos = torch.nonzero(indices == idx_feature, as_tuple=False)

        # If weights are not provided, sample uniformly
        data_weights = torch.ones(len(self.sample_data), dtype=torch.float32)
        if self.weights is not None:
            data_weights = self.weights[:, idx_feature]
        randomly_selected = torch.multinomial(data_weights, num_samples=num_samples, replacement=True)

        samples = []
        for idx_sample in range(num_samples):
            curr_feature_pos = int(feature_pos[idx_sample, 1])
            idx_rand = int(randomly_selected[idx_sample])

            # With/without `idx_feature` set
            input_ids_with, input_ids_without = deepcopy(instance[0]), deepcopy(instance[0])
            for _i in indices[idx_sample, curr_feature_pos + 1:]:
                input_ids_with[_i] = self.sample_data[idx_rand][_i]
                input_ids_without[_i] = self.sample_data[idx_rand][_i]

            input_ids_without[idx_feature] = self.sample_data[idx_rand][idx_feature]

            samples.append(input_ids_with)
            samples.append(input_ids_without)

        # Pad samples to the maximum length of newly-created samples (simplifies handling of special tokens)
        max_sample_len = max([sum(len(curr_subwords) for curr_subwords in curr_sample) for curr_sample in samples])
        sample_input_ids = []
        sample_modeling_kwargs = {k: [] for k in modeling_kwargs}
        for curr_sample in samples:
            curr_input_ids = []
            curr_modeling_kwargs = {k: [] for k in modeling_kwargs}
            for idx_word, curr_subwords in enumerate(curr_sample):
                curr_input_ids.extend(curr_subwords)
                for k in modeling_kwargs:
                    curr_modeling_kwargs[k].extend([modeling_kwargs[k][0, idx_word].item()] * len(curr_subwords))

            sample_input_ids.append(curr_input_ids +
                                    [self.model.tokenizer.pad_token_id] * (max_sample_len - len(curr_input_ids)))
            for k in modeling_kwargs:
                sample_modeling_kwargs[k].append(curr_modeling_kwargs[k] +
                                                 [self.model.tokenizer.pad_token_id] * (max_sample_len - len(curr_modeling_kwargs[k])))

        results = {}
        if self.return_samples:
            results["samples"] = sample_input_ids

        sample_input_ids = torch.tensor(sample_input_ids)
        sample_modeling_kwargs = {k: torch.tensor(v) for k, v in sample_modeling_kwargs.items()}
        scores = self.model.score(sample_input_ids, **sample_modeling_kwargs)
        scores_with = scores[::2]
        scores_without = scores[1::2]
        assert scores_with.shape[0] == scores_without.shape[0]
        diff = scores_with - scores_without

        results.update({
            "diff_mean": torch.mean(diff, dim=0),
            "diff_var": torch.var(diff, dim=0)
        })

        if self.return_scores:
            results["scores"] = scores.tolist()

        return results

    def explain_text(self, text_data: Union[List[str], Tuple[List[str], ...]], label: Optional[int] = 0,
                     min_samples_per_feature: Optional[int] = 100, max_samples: Optional[int] = None,
                     pretokenized_text_data: Optional[Union[List[str], Tuple[List[str], ...]]] = None,  # unused
                     custom_features: Optional[List[List[int]]] = None):
        # Convert instance being interpreted to representation of interpreted model
        model_instance = self.model.words_to_internal([text_data])

        res = self.explain(model_instance["input_ids"], label, perturbable_mask=model_instance["perturbable_mask"],
                           min_samples_per_feature=min_samples_per_feature, max_samples=max_samples,
                           **model_instance["aux_data"])
        res["input"] = model_instance["words"][0]

        return res


class GreedyBaselineIMEExplainer:
    def __init__(self, model: InterpretableModel, return_samples: Optional[bool] = False,
                 return_scores: Optional[bool] = False):
        self.model = model
        self.return_samples = return_samples
        self.return_scores = return_scores

    def explain_text(self, text_data: Union[str, Tuple[str, ...]], label: Optional[int] = 0):
        # Convert instance being interpreted to representation of interpreted model
        model_instance = self.model.to_internal([text_data])
        num_features = model_instance["input_ids"].shape[1]
        perturbable_mask = model_instance["perturbable_mask"][0]
        perturbable_inds = torch.arange(num_features)[perturbable_mask]

        input_ids = model_instance["input_ids"]
        ret_samples = [[] for _ in range(num_features)]
        ret_scores = [[] for _ in range(num_features)]

        importances = torch.zeros(num_features, dtype=torch.float32)
        for idx_feature in perturbable_inds:
            curr_samples = input_ids.repeat((2, 1))  # with/without
            curr_samples[1, idx_feature] = self.model.tokenizer.unk_token_id

            scores = self.model.score(curr_samples, **model_instance["aux_data"])
            diff = scores[0, label] - scores[1, label]
            importances[idx_feature] = diff

            if self.return_samples:
                ret_samples[idx_feature].extend(self.model.from_internal(curr_samples, skip_special_tokens=False, take_as_single_sequence=True))

            if self.return_scores:
                ret_scores[idx_feature].extend(scores.tolist())

        res = {
            "importance": importances,
            "taken_samples": int(perturbable_inds.shape[0])
        }

        if self.return_samples:
            res["samples"] = ret_samples

        if self.return_scores:
            res["scores"] = ret_scores

        return res


if __name__ == "__main__":
    from explain_nlp.methods.modeling import InterpretableBertForSequenceClassification

    model = InterpretableBertForSequenceClassification(
        model_name="/home/matej/Documents/embeddia/interpretability/ime-lm/resources/weights/snli_bert_uncased",
        tokenizer_name="/home/matej/Documents/embeddia/interpretability/ime-lm/resources/weights/snli_bert_uncased",
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
                             return_variance=True,
                             return_num_samples=True,
                             return_samples=True,
                             return_scores=True)

    ex = "The big brown fox jumps over the lazy dog."
    res = explainer.explain_text(ex, label=2, min_samples_per_feature=10, custom_features=[[1], [2], [3, 4], [5, 6, 7, 8, 9], [10]])
    print(res["importance"])

