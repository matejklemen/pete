from typing import Optional, Union, Tuple, List

import numpy as np
import torch
from sklearn.linear_model import Ridge

from explain_nlp.generation.generation_base import SampleGenerator
from explain_nlp.methods.utils import sample_permutations, tensor_indexer, list_indexer
from explain_nlp.modeling.modeling_base import InterpretableModel
from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification


def exponential_kernel(dists: torch.Tensor, kernel_width: float):
    return torch.sqrt(torch.exp(- dists ** 2 / kernel_width ** 2))


class LIMEExplainer:
    def __init__(self, model: InterpretableModel, kernel_width=25.0,
                 return_samples: Optional[bool] = False, return_scores: Optional[bool] = False):
        self.model = model
        self.kernel_width = kernel_width

        self.feature_selector = Ridge(alpha=0.01, fit_intercept=True)
        self.explanation_model = Ridge(alpha=1.0, fit_intercept=True)

        self.return_samples = return_samples
        self.return_scores = return_scores

        self.indexer = tensor_indexer

    def generate_neighbourhood(self, samples: torch.Tensor, removal_mask, **generation_kwargs):
        samples[removal_mask] = self.model.pad_token_id
        return samples

    def explain(self, instance: Union[torch.Tensor, List], label: Optional[int] = 0,
                perturbable_mask: Optional[torch.Tensor] = None, num_samples: Optional[int] = 1000,
                explanation_length: Optional[int] = None, custom_features: Optional[List[List[int]]] = None,
                **modeling_kwargs):
        num_features, num_additional = len(instance[0]), 0
        eff_perturbable_mask = perturbable_mask if perturbable_mask is not None \
            else torch.ones((1, num_features), dtype=torch.bool)

        perturbable_inds = torch.arange(num_features)[eff_perturbable_mask[0]]

        if custom_features is None:
            feature_groups = perturbable_inds
            self.indexer = tensor_indexer

            used_inds = perturbable_inds.tolist()
        else:
            self.indexer = list_indexer
            num_additional = len(custom_features)

            cover_count = torch.zeros(num_features)
            free_features = eff_perturbable_mask.clone()
            feature_groups = []
            for curr_group in custom_features:
                if not torch.all(eff_perturbable_mask[0, curr_group]):
                    raise ValueError(f"At least one of the features in group {curr_group} is not perturbable")
                if torch.any(cover_count[curr_group] > 0):
                    raise ValueError(f"Custom features are not allowed to overlap (feature group {curr_group} overlaps "
                                     f"with some other group)")
                cover_count[curr_group] += 1
                free_features[0, curr_group] = False

                feature_groups.append(curr_group)

            for _, idx_feature in torch.nonzero(free_features, as_tuple=False):
                free_features[0, idx_feature] = False
                feature_groups.append([idx_feature])
                num_additional += 1

            used_inds = list(range(num_features, num_features + num_additional))

        num_used = len(used_inds)

        eff_explanation_length = explanation_length if explanation_length is not None else num_used
        assert eff_explanation_length <= num_used

        permuted_inds = sample_permutations(upper=len(feature_groups),
                                            indices=torch.arange(len(feature_groups)),
                                            num_permutations=num_samples - 1)
        num_removed = torch.randint(1, num_used + 1, size=(num_samples - 1,))

        samples = instance.repeat((num_samples, 1))

        feature_indicators = torch.zeros((num_samples, num_features + num_additional), dtype=torch.bool)
        feature_indicators[:, perturbable_inds] = True  # explained instance
        removed_mask = torch.zeros_like(samples, dtype=torch.bool)
        for idx_sample in range(num_samples - 1):
            curr_removed_features = self.indexer(feature_groups, permuted_inds[idx_sample, :num_removed[idx_sample]])

            removed_mask[idx_sample + 1, curr_removed_features] = True
            feature_indicators[idx_sample + 1, curr_removed_features] = False

        samples = self.generate_neighbourhood(samples, removal_mask=removed_mask, **modeling_kwargs)

        feature_indicators = feature_indicators.float()
        dists = 1.0 - torch.cosine_similarity(feature_indicators[[0]], feature_indicators)
        weights = exponential_kernel(dists, kernel_width=self.kernel_width)

        pred_probas = self.model.score(samples, **modeling_kwargs)

        feature_selector = Ridge(alpha=0.01, fit_intercept=True)
        feature_selector.fit(X=feature_indicators.numpy(),
                             y=pred_probas[:, label].numpy(),
                             sample_weight=weights.numpy())
        coefs = feature_selector.coef_

        sort_indices = np.argsort(-np.abs(coefs))
        used_features = sort_indices[:eff_explanation_length]

        explanation = torch.zeros(num_features + num_additional)
        explanation_model = Ridge(alpha=1.0, fit_intercept=True)
        explanation_model.fit(X=feature_indicators[:, used_features].numpy(),
                              y=pred_probas[:, label].numpy(),
                              sample_weight=weights.numpy())
        explanation[used_features] = torch.tensor(explanation_model.coef_)

        results = {
            "importance": explanation,
            "bias": torch.tensor(explanation_model.intercept_)
        }

        if self.return_samples:
            results["samples"] = samples.tolist()
            results["indicators"] = feature_indicators.tolist()

        if self.return_scores:
            results["scores"] = pred_probas[:, label].tolist()

        if custom_features is not None:
            results["custom_features"] = feature_groups

        return results

    def explain_text(self, text_data: Union[str, Tuple[str, ...]], label: Optional[Union[int, List[int]]] = 0,
                     num_samples: Optional[int] = 1000, explanation_length: Optional[int] = None,
                     pretokenized_text_data: Optional[Union[List[str], Tuple[List[str], ...]]] = None,
                     custom_features: Optional[List[List[int]]] = None):
        # Convert instance being interpreted to representation of interpreted model
        model_instance = self.model.to_internal([text_data],
                                                pretokenized_text_data=[pretokenized_text_data] if pretokenized_text_data is not None else None)

        res = self.explain(model_instance["input_ids"], label, perturbable_mask=model_instance["perturbable_mask"],
                           num_samples=num_samples, explanation_length=explanation_length,
                           custom_features=custom_features, **model_instance["aux_data"])
        res["input"] = self.model.convert_ids_to_tokens(model_instance["input_ids"])[0]

        return res


class LIMEMaskedLMExplainer(LIMEExplainer):
    def __init__(self, model: InterpretableModel, generator: SampleGenerator, kernel_width=25.0,
                 return_samples: Optional[bool] = False, return_scores: Optional[bool] = False):
        super().__init__(model=model, kernel_width=kernel_width,
                         return_samples=return_samples, return_scores=return_scores)
        self.generator = generator

    def generate_neighbourhood(self, samples: torch.Tensor, removal_mask, **generation_kwargs):
        # TODO: control labels!
        generated_examples = self.generator.generate_masked_samples(samples,
                                                                    generation_mask=removal_mask,
                                                                    **generation_kwargs)
        text_examples = self.generator.from_internal(generated_examples, **generation_kwargs)
        model_examples = self.model.to_internal(text_examples)

        return model_examples["input_ids"]

    def transform_to_generator(self, input_ids, perturbable_mask, **modeling_kwargs):
        """ Maps the POSITIONS of perturbable indices in model instance to perturbable indices in generator instance."""
        instance_tokens = self.model.from_internal(input_ids, return_tokens=True, **modeling_kwargs)
        instance_generator = self.generator.to_internal(instance_tokens, is_split_into_units=True)

        model2generator = {}
        for idx_example, alignment_ids in enumerate(instance_generator["aux_data"]["alignment_ids"]):
            for idx_subunit, idx_word in enumerate(alignment_ids):
                if idx_word == -1:
                    continue

                existing_subunits = model2generator.get(idx_word, [])
                existing_subunits.append(idx_subunit)
                model2generator[idx_word] = existing_subunits

        ret = {
            "instance_generator": instance_generator,
            "mapping": model2generator
        }

        if len(modeling_kwargs) > 0:
            ret["instance_generator"]["aux_data"] = modeling_kwargs

        return ret

    def explain(self, instance: Union[torch.Tensor, List], label: Optional[int] = 0,
                perturbable_mask: Optional[torch.Tensor] = None, num_samples: Optional[int] = 1000,
                explanation_length: Optional[int] = None, custom_features: Optional[List[List[int]]] = None,
                **modeling_kwargs):
        num_features, num_additional = len(instance[0]), 0
        eff_perturbable_mask = perturbable_mask if perturbable_mask is not None \
            else torch.ones((1, num_features), dtype=torch.bool)

        perturbable_inds = torch.arange(num_features)[eff_perturbable_mask[0]].tolist()
        perturbable_position = {idx_pert: i for i, idx_pert in enumerate(perturbable_inds)}
        res = self.transform_to_generator(instance, eff_perturbable_mask, **modeling_kwargs)
        mapping = res["mapping"]
        instance_generator = res["instance_generator"]

        if custom_features is None:
            feature_groups, generator_groups = [], []
            has_bigger_units = False
            for idx_feature in perturbable_inds:
                new_feature = mapping[perturbable_position[idx_feature]]
                has_bigger_units |= isinstance(new_feature, list)

                feature_groups.append(idx_feature)
                generator_groups.append(new_feature)

            if has_bigger_units:
                self.indexer = list_indexer
                feature_groups = [[group] for group in feature_groups]
                generator_groups = [[group] if isinstance(group, int) else group for group in generator_groups]
            else:
                self.indexer = tensor_indexer
                feature_groups = torch.tensor(feature_groups)
                generator_groups = torch.tensor(generator_groups)

            used_inds = perturbable_inds
        else:
            self.indexer = list_indexer
            num_additional = len(custom_features)

            cover_count = torch.zeros(num_features)
            free_features = eff_perturbable_mask.clone()
            feature_groups, generator_groups = [], []
            for curr_group in custom_features:
                if not torch.all(eff_perturbable_mask[0, curr_group]):
                    raise ValueError(f"At least one of the features in group {curr_group} is not perturbable")
                if torch.any(cover_count[curr_group] > 0):
                    raise ValueError(f"Custom features are not allowed to overlap (feature group {curr_group} overlaps "
                                     f"with some other group)")
                cover_count[curr_group] += 1
                free_features[0, curr_group] = False

                mapped_group = list(map(lambda idx_feature: mapping[perturbable_position[idx_feature]],
                                        curr_group))
                feature_groups.append(curr_group)
                generator_groups.append(mapped_group)

            for _, idx_feature in torch.nonzero(free_features, as_tuple=False):
                free_features[0, idx_feature] = False
                feature_groups.append([idx_feature])
                generator_groups.append([mapping[perturbable_position[idx_feature]]])
                num_additional += 1

            used_inds = list(range(num_features, num_features + num_additional))

        inds_group = dict(zip(used_inds, range(len(used_inds))))
        num_used = len(used_inds)

        eff_explanation_length = explanation_length if explanation_length is not None else num_used
        assert eff_explanation_length <= num_used

        permuted_inds = sample_permutations(upper=len(feature_groups),
                                            indices=torch.arange(len(feature_groups)),
                                            num_permutations=num_samples - 1)
        num_removed = torch.randint(1, num_used + 1, size=(num_samples - 1,))

        feature_indicators = torch.zeros((num_samples, num_features + num_additional), dtype=torch.bool)
        feature_indicators[:, perturbable_inds] = True  # explained instance

        samples = instance_generator["input_ids"].repeat((num_samples, 1))
        removed_mask = torch.zeros_like(samples, dtype=torch.bool)
        for idx_sample in range(num_samples - 1):
            curr_removed_model = self.indexer(feature_groups, permuted_inds[idx_sample, :num_removed[idx_sample]])
            curr_removed_generator = self.indexer(generator_groups, permuted_inds[idx_sample, :num_removed[idx_sample]])

            removed_mask[idx_sample + 1, curr_removed_generator] = True
            feature_indicators[idx_sample + 1, curr_removed_model] = False

        samples = self.generate_neighbourhood(samples, removal_mask=removed_mask, **modeling_kwargs)

        feature_indicators = feature_indicators.float()
        dists = 1.0 - torch.cosine_similarity(feature_indicators[[0]], feature_indicators)
        weights = exponential_kernel(dists, kernel_width=self.kernel_width)

        pred_probas = self.model.score(samples, **modeling_kwargs)

        feature_selector = Ridge(alpha=0.01, fit_intercept=True)
        feature_selector.fit(X=feature_indicators.numpy(),
                             y=pred_probas[:, label].numpy(),
                             sample_weight=weights.numpy())
        coefs = feature_selector.coef_

        sort_indices = np.argsort(-np.abs(coefs))
        used_features = sort_indices[:eff_explanation_length]

        explanation = torch.zeros(num_features + num_additional)
        explanation_model = Ridge(alpha=1.0, fit_intercept=True)
        explanation_model.fit(X=feature_indicators[:, used_features].numpy(),
                              y=pred_probas[:, label].numpy(),
                              sample_weight=weights.numpy())
        explanation[used_features] = torch.tensor(explanation_model.coef_)

        results = {
            "importance": explanation,
            "bias": torch.tensor(explanation_model.intercept_)
        }

        if self.return_samples:
            results["samples"] = samples.tolist()
            results["indicators"] = feature_indicators.tolist()

        if self.return_scores:
            results["scores"] = pred_probas[:, label].tolist()

        if custom_features is not None:
            results["custom_features"] = feature_groups

        return results


if __name__ == "__main__":
    from explain_nlp.generation.generation_transformers import BertForMaskedLMGenerator
    from explain_nlp.visualizations.highlight import highlight_plot
    from explain_nlp.visualizations.internal import visualize_lime_internals

    model = InterpretableBertForSequenceClassification(
        model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased",
        tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased",
        batch_size=8,
        max_seq_len=41,
        device="cpu"
    )
    # generator = BertForMaskedLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert-base-uncased-snli-mlm",
    #                                      model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert-base-uncased-snli-mlm",
    #                                      max_seq_len=41,
    #                                      batch_size=8,
    #                                      device="cpu",
    #                                      strategy="top_p",
    #                                      top_p=0.95)

    explainer = LIMEExplainer(model, return_samples=True, return_scores=True)
    # explainer = LIMEMaskedLMExplainer(model, generator=generator, return_samples=True, return_scores=True)

    seq = ("A shirtless man skateboards on a ledge.", "A man without a shirt.")
    EXPLAINED_LABEL = 0
    EXPLANATION_LENGTH = 5
    res = explainer.explain_text(seq, label=EXPLAINED_LABEL, num_samples=100)

    visualize_lime_internals(sequence_tokens=res["input"][:19],
                             token_mask=[res["indicators"][_i][:19] for _i in range(len(res["indicators"]))],
                             probabilities=res["scores"],
                             width_per_sample=0.1,
                             height_per_token=0.2,
                             ylabel="Probability (entailment)",
                             sort_key="token_mask")

    highlight_plot([res["input"]],
                   importances=[res["importance"].tolist()],
                   pred_labels=["entailment"],
                   actual_labels=["entailment"],
                   path="tmp_lime.html")
