from typing import Optional, Union, Tuple, List

import numpy as np
import torch
from sklearn.linear_model import Ridge

from explain_nlp.methods.utils import sample_permutations, tensor_indexer, list_indexer
from explain_nlp.modeling.modeling_base import InterpretableModel
from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification
from explain_nlp.visualizations.highlight import highlight_plot


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

    def explain(self, instance: Union[torch.Tensor, List], label: Optional[int] = 0,
                perturbable_mask: Optional[torch.Tensor] = None, num_samples: Optional[int] = 1000,
                explanation_length: Optional[int] = None, custom_features: Optional[List[List[int]]] = None,
                **modeling_kwargs):
        num_features = len(instance[0])
        num_additional = 0
        eff_perturbable_mask = perturbable_mask if perturbable_mask is not None \
            else torch.ones((1, num_features), dtype=torch.bool)
        self.indexer = tensor_indexer

        # Contains all primary features of instance OR all primary features of instance + custom features, where
        # custom features are lists of primary features (e.g. corresponding to sentences).
        superfeatures = torch.arange(num_features)  # type: Union[torch.Tensor, list]
        if custom_features is not None:
            superfeatures = list(range(num_features))
            self.indexer = list_indexer
            num_additional = len(custom_features)

            cover_count = torch.zeros(num_features)
            free_features = eff_perturbable_mask.clone()
            for curr_group in custom_features:
                superfeatures.append(curr_group)
                free_features[0, curr_group] = False
                if not torch.all(eff_perturbable_mask[0, curr_group]):
                    raise ValueError(f"At least one of the features in group {curr_group} is not perturbable")
                if torch.any(cover_count[curr_group] > 0):
                    raise ValueError(f"Custom features are not allowed to overlap (feature group {curr_group} overlaps "
                                     f"with some other group)")
                cover_count[curr_group] += 1

            for _, idx_feature in torch.nonzero(free_features, as_tuple=False):
                free_features[0, idx_feature] = False
                superfeatures.append([idx_feature.item()])
                num_additional += 1

        perturbable_inds = torch.arange(num_features)[eff_perturbable_mask[0]] \
            if custom_features is None else torch.arange(num_features, num_features + num_additional)
        eff_explanation_length = explanation_length if explanation_length is not None else perturbable_inds.shape[0]
        assert eff_explanation_length <= perturbable_inds.shape[0]

        permuted_inds = sample_permutations(upper=(num_features + num_additional),
                                            indices=perturbable_inds,
                                            num_permutations=num_samples - 1)
        num_removed = torch.randint(1, len(perturbable_inds) + 1, size=(num_samples - 1,))
        samples = instance.repeat((num_samples, 1))
        # 0 = delete feature (= set value to PAD), 1 = take original feature value
        feature_indicators = torch.zeros((num_samples, num_features + num_additional))
        feature_indicators[0, perturbable_inds] = 1.0  # explained instance
        for idx_sample in range(num_samples - 1):
            curr_kept = permuted_inds[idx_sample, num_removed[idx_sample]:]
            feature_indicators[idx_sample + 1, curr_kept] = 1.0

            if num_additional > 0:
                curr_removed_features = self.indexer(superfeatures, permuted_inds[idx_sample, :num_removed[idx_sample]])
            else:
                curr_removed_features = self.indexer(superfeatures, permuted_inds[idx_sample, :num_removed[idx_sample]])

            samples[idx_sample + 1, curr_removed_features] = self.model.pad_token_id

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
            results["scores"] = pred_probas[:, [label]]

        if custom_features is not None:
            results["custom_features"] = superfeatures[num_features:]

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


if __name__ == "__main__":
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
    EXPLANATION_LENGTH = 5
    res = explainer.explain_text(seq, label=EXPLAINED_LABEL, num_samples=3, explanation_length=EXPLANATION_LENGTH,
                                 custom_features=[[1], [2, 3], [4], [5, 6], [7], [8], [9], [10],
                                                  [12], [13], [14], [15], [16], [17]])
    print(res)

    highlight_plot([res["input"]],
                   importances=[res["importance"].tolist()],
                   pred_labels=["entailment"],
                   actual_labels=["entailment"],
                   custom_features=[res.get("custom_features")],
                   path="tmp_lime.html")
