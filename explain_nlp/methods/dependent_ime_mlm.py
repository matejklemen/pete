from typing import Optional, Union, List

import torch

from explain_nlp.generation.generation_base import SampleGenerator
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.methods.utils import sample_permutations
from explain_nlp.modeling.modeling_base import InterpretableModel

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DependentIMEMaskedLMExplainer(IMEExplainer):
    def __init__(self, model: InterpretableModel, generator: SampleGenerator,
                 confidence_interval: Optional[float] = None,  max_abs_error: Optional[float] = None,
                 return_variance: Optional[bool] = False, return_num_samples: Optional[bool] = False,
                 return_samples: Optional[bool] = False, return_scores: Optional[bool] = False,
                 criterion: Optional[str] = "squared_error"):
        dummy_sample_data = torch.randint(5, (1, 1), dtype=torch.long)
        super().__init__(sample_data=dummy_sample_data, model=model, confidence_interval=confidence_interval,
                         max_abs_error=max_abs_error, return_variance=return_variance,
                         return_num_samples=return_num_samples, return_samples=return_samples,
                         return_scores=return_scores, criterion=criterion)

        self.generator = generator

    def get_generator_mapping(self, input_ids, perturbable_mask, **modeling_kwargs):
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

        return model2generator

    @torch.no_grad()
    def estimate_feature_importance(self, idx_feature: int, instance: torch.Tensor,
                                    num_samples: int, perturbable_mask: torch.Tensor,
                                    feature_groups: Optional[Union[torch.Tensor, List[List[int]]]] = None,
                                    **modeling_kwargs):
        # Note: instance is currently supposed to be of shape [1, num_features]
        num_features = int(len(instance[0]))

        if feature_groups is None:
            eff_feature_groups = torch.arange(num_features)[perturbable_mask[0]]
            idx_superfeature = eff_feature_groups.tolist().index(idx_feature)
        else:
            eff_feature_groups = feature_groups
            idx_superfeature = idx_feature

        if hasattr(self.generator, "label_weights"):
            randomly_selected_label = torch.multinomial(self.generator.label_weights, num_samples=num_samples, replacement=True)
            randomly_selected_label = [self.generator.control_labels_str[i] for i in randomly_selected_label]
        else:
            randomly_selected_label = [None] * num_samples

        # TODO: get generator instance provided as argument (in place of `instance`)
        instance_tokens = self.model.from_internal(instance, return_tokens=True, **modeling_kwargs)
        instance_generator = self.generator.to_internal(instance_tokens, is_split_into_units=True)
        num_gen_features = instance_generator["input_ids"].shape[1]
        est_instance_features = eff_feature_groups[idx_feature]

        # Permuted POSITIONS of (super)features inside `eff_feature_groups`
        indices = sample_permutations(upper=len(eff_feature_groups),
                                      indices=torch.arange(len(eff_feature_groups)),
                                      num_permutations=num_samples)
        feature_pos = torch.nonzero(indices == idx_superfeature, as_tuple=False)

        is_masked = torch.zeros((2 * num_samples, num_gen_features), dtype=torch.bool)
        for idx_sample in range(num_samples):
            curr_feature_pos = int(feature_pos[idx_sample, 1])
            changed_features = self.indexer(eff_feature_groups, indices[idx_sample, curr_feature_pos + 1:])

            # 1 sample with, 1 sample without current feature fixed
            is_masked[2 * idx_sample: 2 * idx_sample + 2, changed_features] = True
            is_masked[2 * idx_sample + 1, est_instance_features] = True

        all_examples = self.generator.generate_masked_samples(instance_generator["input_ids"],
                                                              generation_mask=is_masked,
                                                              control_labels=randomly_selected_label,
                                                              **instance_generator["aux_data"])

        text_examples = self.generator.from_internal(all_examples, **instance_generator["aux_data"])
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
    from explain_nlp.generation.generation_transformers import BertForMaskedLMGenerator, RobertaForMaskedLMGenerator
    from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification

    model = InterpretableBertForSequenceClassification(tokenizer_name="bert-base-uncased",
                                                       model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased",
                                                       batch_size=2,
                                                       device="cpu",
                                                       max_seq_len=41)
    generator = BertForMaskedLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert-base-uncased-snli-mlm",
                                         model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert-base-uncased-snli-mlm",
                                         batch_size=10,
                                         max_seq_len=41,
                                         device="cpu",
                                         strategy="top_p",
                                         top_p=0.95)

    explainer = DependentIMEMaskedLMExplainer(model=model,
                                              generator=generator,
                                              return_samples=True,
                                              return_scores=True,
                                              return_variance=True,
                                              return_num_samples=True)

    seq = ("A shirtless man skateboards on a ledge.", "A man without a shirt")
    res = explainer.explain_text(seq, label=0, min_samples_per_feature=5)
    print(f"Sum of importances: {sum(res['importance'])}")
    for curr_token, curr_imp, curr_var in zip(res["input"], res["importance"], res["var"]):
        print(f"{curr_token} = {curr_imp: .4f} (var: {curr_var: .4f})")
