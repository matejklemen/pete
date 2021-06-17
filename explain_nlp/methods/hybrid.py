from typing import Optional, Union, List

import torch

from explain_nlp.experimental.data import load_nli
from explain_nlp.generation.generation_base import SampleGenerator
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.modeling.modeling_base import InterpretableModel
from explain_nlp.methods.utils import sample_permutations
from explain_nlp.utils import EncodingException


def create_uniform_weights(input_ids, special_tokens_mask):
    """ Creates weight matrix such that valid tokens (i.e. not special) all get weight 1.0 and the others 0.0.
        Returns matrix with shape like `input_ids`. """
    weights = torch.ones_like(input_ids, dtype=torch.float32)
    weights[special_tokens_mask.bool()] = 0.0
    return weights


class HybridIMEExplainer(IMEExplainer):
    def __init__(self, sample_data_generator: torch.Tensor, model: InterpretableModel, generator: SampleGenerator,
                 data_weights: Optional[torch.Tensor] = None,
                 confidence_interval: Optional[float] = None, max_abs_error: Optional[float] = None,
                 return_num_samples: Optional[bool] = False,
                 return_samples: Optional[bool] = False, return_scores: Optional[bool] = False,
                 criterion: Optional[str] = "squared_error", shared_vocabulary: Optional[bool] = False):
        super().__init__(sample_data=sample_data_generator, model=model, data_weights=data_weights,
                         confidence_interval=confidence_interval, max_abs_error=max_abs_error,
                         return_num_samples=return_num_samples, return_samples=return_samples,
                         return_scores=return_scores, criterion=criterion, shared_vocabulary=shared_vocabulary)
        self.generator = generator
        self.feature_varies = None

        self.update_sample_data(sample_data_generator, data_weights=data_weights)

    def update_sample_data(self, new_data: torch.Tensor, data_weights: Optional[torch.FloatTensor] = None):
        super().update_sample_data(new_data=new_data, data_weights=data_weights)
        if self.weights is None:
            self.weights = torch.ones_like(self.sample_data, dtype=torch.float32)

        self.feature_varies = torch.gt(torch.sum(self.weights, dim=0), (0.0 + 1e-6))

    def model_to_generator(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor,
                           **modeling_kwargs):
        instance_tokens = self.model.from_internal(input_ids, return_tokens=True, **modeling_kwargs)
        try:
            instance_generator = self.generator.to_internal(instance_tokens, is_split_into_units=True,
                                                            allow_truncation=False)
        except EncodingException:
            raise ValueError("Conversion between model instance and generator's instance could not be performed: "
                             "the obtained generator instance is longer than allowed generator's maximum length.\n"
                             "To fix this, either (1) increase generator's max_seq_len or (2) decrease model's "
                             "max_seq_len.")

        model2generator = {}
        for idx_example, alignment_ids in enumerate(instance_generator["aux_data"]["alignment_ids"]):
            for idx_subunit, idx_word in enumerate(alignment_ids):
                if idx_word == -1:
                    continue

                existing_subunits = model2generator.get(idx_word, [])
                existing_subunits.append(idx_subunit)
                model2generator[idx_word] = existing_subunits

        return {
            "generator_instance": instance_generator,
            "mapping": model2generator
        }

    def estimate_feature_importance(self, idx_feature: int, instance: torch.Tensor,
                                    num_samples: int, perturbable_mask: torch.Tensor,
                                    feature_groups: Optional[Union[torch.Tensor, List[List[int]]]] = None,
                                    **generation_kwargs):
        # Note: instance is currently supposed to be of shape [1, num_features]
        num_features = int(len(instance[0]))

        if feature_groups is None:
            eff_feature_groups = torch.arange(num_features)[perturbable_mask[0]].tolist()
            idx_superfeature = eff_feature_groups.index(idx_feature)
            eff_feature_groups = [[_i] for _i in eff_feature_groups]
        else:
            eff_feature_groups = feature_groups
            idx_superfeature = idx_feature

        if hasattr(self.generator, "label_weights"):
            randomly_selected_label = torch.multinomial(self.generator.label_weights, num_samples=num_samples,
                                                        replacement=True)
            # A pair belonging to same sample is assigned the same label
            randomly_selected_label = torch.stack((randomly_selected_label, randomly_selected_label)).T.flatten()
            randomly_selected_label = [self.generator.control_labels_str[i] for i in randomly_selected_label]
        else:
            randomly_selected_label = [None] * (2 * num_samples)

        est_instance_features = eff_feature_groups[idx_superfeature]
        print([self.generator.tokenizer.decode(curr_id) for curr_id in instance[0, est_instance_features]])

        # Permuted POSITIONS of (super)features inside `eff_feature_groups`
        indices = sample_permutations(upper=len(eff_feature_groups),
                                      indices=torch.arange(len(eff_feature_groups)),
                                      num_permutations=num_samples)
        feature_pos = torch.nonzero(indices == idx_superfeature, as_tuple=False)

        data_weights = self.weights[:, est_instance_features]
        if data_weights.dim() > 1:
            # weight = 1 if at least one token is non-special
            data_weights = torch.gt(torch.sum(data_weights, dim=1), 0 + 1e-6).float()

        if torch.any(self.feature_varies[est_instance_features]):
            rand_idx = torch.multinomial(data_weights, num_samples=num_samples, replacement=True).unsqueeze(1)
            randomly_selected_val = self.sample_data[rand_idx, est_instance_features]
        else:
            randomly_selected_val = instance[0, est_instance_features].repeat((num_samples, 1))

        is_masked = torch.zeros((2 * num_samples, num_features), dtype=torch.bool)
        samples = instance.repeat((2 * num_samples, 1))
        for idx_sample in range(num_samples):
            curr_feature_pos = int(feature_pos[idx_sample, 1])
            changed_features = self.indexer(eff_feature_groups, indices[idx_sample, curr_feature_pos + 1:])

            # 1 sample with, 1 sample "without" current feature fixed:
            # current feature randomized in second sample, but using sample data instead of a generator
            is_masked[2 * idx_sample: 2 * idx_sample + 2, changed_features] = True
            samples[2 * idx_sample + 1, est_instance_features] = randomly_selected_val[idx_sample]

        all_examples = self.generator.generate_masked_samples(samples,
                                                              generation_mask=is_masked,
                                                              control_labels=randomly_selected_label,
                                                              **generation_kwargs)

        text_examples = self.generator.from_internal(all_examples, **generation_kwargs)
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
    from explain_nlp.generation.generation_transformers import BertForMaskedLMGenerator, RobertaForMaskedLMGenerator

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
                                         top_p=0.01,
                                         monte_carlo_dropout=False)

    df_data = load_nli("/home/matej/Documents/data/snli/snli_1.0_dev.txt")
    data = generator.to_internal(list(zip(df_data["sentence1"].values, df_data["sentence2"].values)))
    weights = create_uniform_weights(data["input_ids"], torch.logical_not(data["perturbable_mask"]))

    explainer = HybridIMEExplainer(model=model, generator=generator,
                                   sample_data_generator=data["input_ids"],
                                   data_weights=weights,
                                   return_num_samples=True,
                                   return_samples=True,
                                   return_scores=True)

    ex = ("A patient is being worked on by doctors and nurses.", "A man is sleeping.")
    res = explainer.explain_text(ex, label=2, min_samples_per_feature=5)
    print(f"Sum:")
    print(torch.sum(res["importance"]))
    for curr_token, curr_imp, curr_var in zip(res["input"], res["importance"], res["var"]):
        print(f"{curr_token} = {curr_imp: .4f} (var: {curr_var: .4f})")

