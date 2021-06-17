from typing import Optional, Union, Tuple, List

import numpy as np
import torch
from sklearn.linear_model import Ridge

from explain_nlp.methods.utils import sample_permutations, list_indexer, handle_custom_features
from explain_nlp.modeling.modeling_base import InterpretableModel
from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification


def exponential_kernel(dists: torch.Tensor, kernel_width: float):
    return torch.sqrt(torch.exp(- dists ** 2 / kernel_width ** 2))


class LIMEExplainer:
    def __init__(self, model: InterpretableModel, kernel_width=1.0,
                 return_samples: Optional[bool] = False, return_scores: Optional[bool] = False,
                 return_metrics: Optional[bool] = True, shared_vocabulary: Optional[bool] = False):
        self.model = model
        self.kernel_width = kernel_width

        self.feature_selector = Ridge(alpha=0.01, fit_intercept=True)
        self.explanation_model = Ridge(alpha=1.0, fit_intercept=True)

        self.return_samples = return_samples
        self.return_scores = return_scores
        self.return_metrics = return_metrics
        self.shared_vocabulary = shared_vocabulary

        self.indexer = list_indexer

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

        # In LIME and LIME+LM where model and generator use same vocabulary, no conversion of data is needed
        if self.shared_vocabulary:
            conversion_data = LIMEExplainer.model_to_generator(self, instance, eff_perturbable_mask, **modeling_kwargs)
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

        num_used = len(used_inds)
        used_inds = torch.tensor(used_inds)

        eff_explanation_length = explanation_length if explanation_length is not None else num_used
        assert eff_explanation_length <= num_used

        permuted_inds = sample_permutations(upper=len(feature_groups),
                                            indices=torch.arange(len(feature_groups)),
                                            num_permutations=num_samples - 1)
        num_removed = torch.randint(1, num_used + 1, size=(num_samples - 1,))

        samples = generator_instance["input_ids"].repeat((num_samples, 1))

        feature_indicators = torch.zeros((num_samples, num_features + num_additional), dtype=torch.bool)
        feature_indicators[:, used_inds] = True  # explained instance

        removed_mask = torch.zeros_like(samples, dtype=torch.bool)
        for idx_sample in range(num_samples - 1):
            groups_to_remove = permuted_inds[idx_sample, :num_removed[idx_sample]]
            curr_removed_features = self.indexer(feature_groups, groups_to_remove)

            removed_mask[idx_sample + 1, curr_removed_features] = True
            feature_indicators[idx_sample + 1, used_inds[groups_to_remove]] = False

        samples = self.generate_neighbourhood(samples, removal_mask=removed_mask, **generator_instance["aux_data"])

        feature_indicators = feature_indicators.float()
        dists = 1.0 - torch.cosine_similarity(feature_indicators[[0]], feature_indicators)
        weights = exponential_kernel(dists, kernel_width=self.kernel_width)

        pred_probas = self.model.score(samples, **modeling_kwargs)

        np_indicators = feature_indicators.numpy()
        np_probas = pred_probas.numpy()
        np_weights = weights.numpy()

        feature_selector = Ridge(alpha=0.01, fit_intercept=True)
        feature_selector.fit(X=np_indicators,
                             y=np_probas[:, label],
                             sample_weight=np_weights)
        coefs = feature_selector.coef_

        sort_indices = np.argsort(-np.abs(coefs))
        used_features = sort_indices[:eff_explanation_length]

        explanation = torch.zeros(num_features + num_additional)
        explanation_model = Ridge(alpha=1.0, fit_intercept=True)
        explanation_model.fit(X=np_indicators[:, used_features],
                              y=np_probas[:, label],
                              sample_weight=np_weights)
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

        if self.return_metrics:
            results["pred_model"] = np_probas[0, label]
            results["pred_surrogate"] = explanation_model.predict(np_indicators[0: 1, used_features])[0]
            results["pred_mean"] = np.mean(np_probas)
            results["pred_median"] = np.median(np_probas)

        if custom_features is not None:
            results["custom_features"] = feature_groups

        return results

    def explain_text(self, text_data: Union[str, Tuple[str, ...]], label: Optional[Union[int, List[int]]] = 0,
                     num_samples: Optional[int] = 1000, explanation_length: Optional[int] = None,
                     pretokenized_text_data: Optional[Union[List[str], Tuple[List[str], ...]]] = None,
                     custom_features: Optional[List[List[int]]] = None):
        # Convert instance being interpreted to representation of interpreted model
        is_split_into_units = pretokenized_text_data is not None
        model_instance = self.model.to_internal([pretokenized_text_data if is_split_into_units else text_data],
                                                is_split_into_units=is_split_into_units)

        res = self.explain(model_instance["input_ids"], label, perturbable_mask=model_instance["perturbable_mask"],
                           num_samples=num_samples, explanation_length=explanation_length,
                           custom_features=custom_features, **model_instance["aux_data"])
        res["input"] = self.model.from_internal(model_instance["input_ids"],
                                                take_as_single_sequence=True,
                                                skip_special_tokens=False,
                                                return_tokens=True)[0]
        res["taken_samples"] = num_samples

        return res


if __name__ == "__main__":
    from explain_nlp.visualizations.highlight import highlight_plot
    from explain_nlp.visualizations.internal import visualize_lime_internals

    model = InterpretableBertForSequenceClassification(
        model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased",
        tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased",
        batch_size=8,
        max_seq_len=41,
        device="cpu"
    )

    explainer = LIMEExplainer(model, return_samples=True, return_scores=True)

    seq = ("A shirtless man skateboards on a ledge.", "A man without a shirt.")
    EXPLAINED_LABEL = 0
    EXPLANATION_LENGTH = None
    res = explainer.explain_text(seq, label=EXPLAINED_LABEL, num_samples=100, explanation_length=EXPLANATION_LENGTH)

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
