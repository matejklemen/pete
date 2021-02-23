from typing import Optional, Union, List

import torch

from explain_nlp.experimental.data import load_nli
from explain_nlp.generation.generation_transformers import BertForMaskedLMGenerator
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.modeling.modeling_base import InterpretableModel
from explain_nlp.methods.utils import sample_permutations


def create_uniform_weights(input_ids, special_tokens_mask):
    """ Creates weight matrix such that valid tokens (i.e. not special) all get weight 1.0 and the others 0.0.
        Returns matrix with shape like `input_ids`. """
    weights = torch.ones_like(input_ids, dtype=torch.float32)
    weights[special_tokens_mask.bool()] = 0.0
    return weights


class HybridIMEExplainer(IMEExplainer):
    def __init__(self, sample_data: torch.Tensor, model: InterpretableModel, generator: BertForMaskedLMGenerator,
                 data_weights: Optional[torch.Tensor] = None,
                 confidence_interval: Optional[float] = None, max_abs_error: Optional[float] = None,
                 return_variance: Optional[bool] = False, return_num_samples: Optional[bool] = False,
                 return_samples: Optional[bool] = False, return_scores: Optional[bool] = False,
                 criterion: Optional[str] = "squared_error"):
        super().__init__(sample_data=sample_data, model=model, data_weights=data_weights,
                         confidence_interval=confidence_interval, max_abs_error=max_abs_error,
                         return_variance=return_variance, return_num_samples=return_num_samples,
                         return_samples=return_samples, return_scores=return_scores,
                         criterion=criterion)
        self.generator = generator
        self.feature_varies = None

        self.update_sample_data(sample_data, data_weights=data_weights)

    def update_sample_data(self, new_data: torch.Tensor, data_weights: Optional[torch.FloatTensor] = None):
        self.sample_data = new_data
        self.weights = data_weights
        self.num_features = new_data.shape[1]

        if self.weights is None:
            self.weights = torch.ones_like(self.sample_data, dtype=torch.float32)

        self.feature_varies = torch.gt(torch.sum(self.weights, dim=0), (0.0 + 1e-6))

    @torch.no_grad()
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

        if feature_groups is not None:
            eff_feature_groups = feature_groups
            idx_superfeature = feature_groups.index(idx_feature)
        else:
            eff_feature_groups = torch.arange(num_features)[perturbable_mask[0]]
            idx_superfeature = eff_feature_groups.tolist().index(idx_feature)

        # Permuted POSITIONS of (super)features inside `eff_feature_groups`
        indices = sample_permutations(upper=len(eff_feature_groups),
                                      indices=torch.arange(len(eff_feature_groups)),
                                      num_permutations=num_samples)
        feature_pos = torch.nonzero(indices == idx_superfeature, as_tuple=False)

        data_weights = self.weights[:, idx_feature]
        if data_weights.dim() > 1:
            # weight = 1 if at least one token is non-special
            data_weights = torch.gt(torch.sum(data_weights, dim=1), 0 + 1e-6).float()

        # TODO: if a feature does not vary, we can return preemptively (other randomized features are identical)
        if torch.any(self.feature_varies[idx_feature]):
            rand_idx = torch.multinomial(data_weights, num_samples=num_samples, replacement=True).unsqueeze(1)
            randomly_selected_val = self.sample_data[rand_idx, idx_feature]
        else:
            randomly_selected_val = instance[0, idx_feature].repeat((num_samples, 1))

        if hasattr(self.generator, "label_weights"):
            randomly_selected_label = torch.multinomial(self.generator.label_weights, num_samples=num_samples, replacement=True)
            randomly_selected_label = [self.generator.control_labels_str[i] for i in randomly_selected_label]
        else:
            randomly_selected_label = [None] * num_samples

        # is_masked[i] = perturbed features for i-th sample (without observed feature)
        masked_samples = instance.repeat((2 * num_samples, 1))
        for idx_sample in range(num_samples):
            curr_feature_pos = int(feature_pos[idx_sample, 1])
            changed_features = self.indexer(eff_feature_groups, indices[idx_sample, curr_feature_pos + 1:])

            # 1 sample with, 1 sample "without" current feature fixed:
            # current feature randomized in second sample, but using sample data instead of a generator
            masked_samples[2 * idx_sample: 2 * idx_sample + 2, changed_features] = self.model.mask_token_id
            masked_samples[2 * idx_sample + 1, idx_feature] = randomly_selected_val[idx_sample]

        text_masked_samples = []
        for seq_or_seq_pair in self.model.from_internal(masked_samples, skip_special_tokens=False, **modeling_kwargs):
            if isinstance(seq_or_seq_pair, tuple):
                s1, s2 = seq_or_seq_pair
                text_masked_samples.append((s1.replace(self.model.mask_token, self.generator.mask_token),
                                            s2.replace(self.model.mask_token, self.generator.mask_token)))
            else:
                text_masked_samples.append(seq_or_seq_pair.replace(self.model.mask_token, self.generator.mask_token))

        instances_generator = self.generator.to_internal(text_masked_samples)
        generated_examples = self.generator.generate_masked_samples(instances_generator["input_ids"],
                                                                    control_labels=randomly_selected_label,
                                                                    **instances_generator["aux_data"])

        text_examples = self.generator.from_internal(generated_examples)
        model_examples = self.model.to_internal(text_examples)

        scores = self.model.score(model_examples["input_ids"], **model_examples["aux_data"])
        scores_with = scores[::2]
        scores_without = scores[1::2]
        assert scores_with.shape[0] == scores_without.shape[0]
        diff = scores_with - scores_without

        results = {
            "diff_mean": torch.mean(diff, dim=0),
            "diff_var": torch.var(diff, dim=0)
        }

        if self.return_samples:
            results["samples"] = model_examples["input_ids"].tolist()

        if self.return_scores:
            results["scores"] = scores.tolist()

        return results


if __name__ == "__main__":
    from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification

    model = InterpretableBertForSequenceClassification(
        model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased",
        tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased",
        batch_size=2,
        max_seq_len=41,
        device="cpu"
    )
    generator = BertForMaskedLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert-base-uncased-snli-mlm",
                                         model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert-base-uncased-snli-mlm",
                                         batch_size=10,
                                         max_seq_len=41,
                                         device="cpu",
                                         strategy="top_p",
                                         top_p=0.05,
                                         monte_carlo_dropout=False)

    df_data = load_nli("/home/matej/Documents/data/snli/snli_1.0_dev.txt")
    data = model.to_internal([(s1, s2) for s1, s2 in df_data[["sentence1", "sentence2"]].values])
    weights = create_uniform_weights(data["input_ids"], torch.logical_not(data["perturbable_mask"]))

    explainer = HybridIMEExplainer(model=model, generator=generator,
                                   sample_data=data["input_ids"],
                                   data_weights=weights,
                                   return_variance=True,
                                   return_num_samples=True,
                                   return_samples=True,
                                   return_scores=True,
                                   is_aligned_vocabulary=True)

    ex = ("A patient is being worked on by doctors and nurses.", "A man is sleeping.")
    res = explainer.explain_text(ex, label=2, min_samples_per_feature=10)
    for curr_token, curr_imp, curr_var in zip(res["input"], res["importance"], res["var"]):
        print(f"{curr_token} = {curr_imp: .4f} (var: {curr_var: .4f})")
