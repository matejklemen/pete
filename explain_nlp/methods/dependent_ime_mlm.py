from typing import Optional, Union, List

import torch

from explain_nlp.generation.generation_base import SampleGenerator
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.methods.utils import sample_permutations
from explain_nlp.modeling.modeling_base import InterpretableModel
from explain_nlp.utils import EncodingException

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DependentIMEMaskedLMExplainer(IMEExplainer):
    def __init__(self, model: InterpretableModel, generator: SampleGenerator,
                 confidence_interval: Optional[float] = None,  max_abs_error: Optional[float] = None,
                 return_num_samples: Optional[bool] = False, return_samples: Optional[bool] = False,
                 return_scores: Optional[bool] = False, criterion: Optional[str] = "squared_error",
                 shared_vocabulary: Optional[bool] = False):
        dummy_sample_data = torch.randint(5, (1, 1), dtype=torch.long)
        super().__init__(sample_data=dummy_sample_data, model=model, confidence_interval=confidence_interval,
                         max_abs_error=max_abs_error, return_num_samples=return_num_samples,
                         return_samples=return_samples, return_scores=return_scores, criterion=criterion,
                         shared_vocabulary=shared_vocabulary)

        self.generator = generator

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

    @torch.no_grad()
    def estimate_feature_importance(self, idx_feature: int, instance: torch.Tensor,
                                    num_samples: int, perturbable_mask: torch.Tensor,
                                    feature_groups: Optional[Union[torch.Tensor, List[List[int]]]] = None,
                                    **generation_kwargs):
        """ Estimate importance of a single feature or a group of features for `instance` using `num_samples` samples,
        where each sample corresponds to a pair of perturbations (one with estimated feature set and another
        with estimated feature randomized).


        IMPORTANT: `instance`, `perturbable_mask` and `**generation_kwargs` should all be data in generator's
        representation, as opposed to how it is in IMEExplainer, where the input is actually in model's representation.

        Args:
            idx_feature:
                Feature whose importance is estimated. If `feature_groups` is provided, `idx_feature` points to the
                position of the estimated custom feature. For example, idx_feature=1 and feature_groups=[[1], [2]]
                means that the importance of feature 2 is estimated
            instance:
                Explained instance, shape: [1, num_features].
            num_samples:
                Number of samples to take.
            perturbable_mask:
                Mask of features that can be perturbed ("modified"), shape: [1, num_features].
            feature_groups:
                Groups that define which features are to be taken as an atomic unit (are to be perturbed together).
                If not provided, groups of single perturbable features are used.
            **generation_kwargs:
                Additional generation data (e.g. attention masks,...)
        """
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

        # Permuted POSITIONS of (super)features inside `eff_feature_groups`
        indices = sample_permutations(upper=len(eff_feature_groups),
                                      indices=torch.arange(len(eff_feature_groups)),
                                      num_permutations=num_samples)
        feature_pos = torch.nonzero(indices == idx_superfeature, as_tuple=False)

        is_masked = torch.zeros((2 * num_samples, num_features), dtype=torch.bool)
        for idx_sample in range(num_samples):
            curr_feature_pos = int(feature_pos[idx_sample, 1])
            changed_features = self.indexer(eff_feature_groups, indices[idx_sample, curr_feature_pos + 1:])

            # 1 sample with, 1 sample without current feature fixed
            is_masked[2 * idx_sample: 2 * idx_sample + 2, changed_features] = True
            is_masked[2 * idx_sample + 1, est_instance_features] = True

        all_examples = self.generator.generate_masked_samples(instance,
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
    from explain_nlp.generation.generation_transformers import BertForMaskedLMGenerator
    from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification

    model = InterpretableBertForSequenceClassification(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased",
                                                       model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased",
                                                       batch_size=2,
                                                       device="cpu",
                                                       max_seq_len=41)

    generator = BertForMaskedLMGenerator(
        tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert-base-uncased-snli-mlm",
        model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert-base-uncased-snli-mlm",
        batch_size=10,
        max_seq_len=41,
        device="cpu",
        strategy="top_p",
        top_p=0.95
    )

    explainer = DependentIMEMaskedLMExplainer(model=model,
                                              generator=generator,
                                              return_samples=True,
                                              return_scores=True,
                                              return_num_samples=True)

    seq = ("A shirtless man skateboards on a ledge.", "A man without a shirt")
    res = explainer.explain_text(seq, label=0, min_samples_per_feature=5)
    for curr_token, curr_imp, curr_var in zip(res["input"], res["importance"], res["var"]):
        print(f"{curr_token} = {curr_imp: .4f} (var: {curr_var: .4f})")
