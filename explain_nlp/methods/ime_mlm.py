from typing import Optional, Union, Tuple

import torch

from explain_nlp.methods.generation import SampleGenerator, BertForMaskedLMGenerator
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.methods.modeling import InterpretableModel, InterpretableBertForSequenceClassification
from explain_nlp.methods.utils import sample_permutations

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class IMEMaskedLMExplainer(IMEExplainer):
    def __init__(self, model: InterpretableModel, generator: SampleGenerator,
                 confidence_interval: Optional[float] = None,  max_abs_error: Optional[float] = None,
                 num_generated_samples: Optional[int] = 10, return_variance: Optional[bool] = False,
                 return_num_samples: Optional[bool] = False, return_samples: Optional[bool] = False,
                 return_scores: Optional[bool] = False):
        # IME requires sampling data so we give it dummy data and later override it with generated data
        dummy_sample_data = torch.empty((0, 0), dtype=torch.long)
        super().__init__(sample_data=dummy_sample_data, model=model, confidence_interval=confidence_interval,
                         max_abs_error=max_abs_error, return_variance=return_variance,
                         return_num_samples=return_num_samples, return_samples=return_samples,
                         return_scores=return_scores)

        self.generator = generator
        self.num_generated_samples = num_generated_samples

        # valid_indices[i] contains indices of examples which have the token `i` different from explained instance
        #  OR all indices, if there are no such examples - should be overwritten for every instance
        self.valid_indices = []

    def estimate_feature_importance(self, idx_feature: int, instance: torch.Tensor, num_samples: int,
                                    perturbable_mask: torch.Tensor, label: Optional[str] = None, **modeling_kwargs):
        # Note: instance is currently supposed to be of shape [1, num_features]
        num_features = int(instance.shape[1])
        perturbable_inds = torch.arange(num_features)[perturbable_mask[0]]

        if num_features != self.num_features:
            raise ValueError(f"Number of features in instance ({num_features}) "
                             f"does not match number of features in sampling data ({self.num_features})")

        indices = sample_permutations(upper=num_features, indices=perturbable_inds,
                                      num_permutations=num_samples)
        feature_pos = torch.nonzero(indices == idx_feature, as_tuple=False)

        samples = instance.repeat((2 * num_samples, 1))
        for idx_sample in range(num_samples):
            curr_feature_pos = int(feature_pos[idx_sample, 1])
            idx_rand = int(torch.randint(len(self.valid_indices[idx_feature]), size=()))
            idx_rand = self.valid_indices[idx_feature][idx_rand]

            # With feature `idx_feature` set
            samples[2 * idx_sample, indices[idx_sample, curr_feature_pos + 1:]] = \
                self.sample_data[idx_rand, indices[idx_sample, curr_feature_pos + 1:]]

            # With feature `idx_feature` randomized
            samples[2 * idx_sample + 1, indices[idx_sample, curr_feature_pos:]] = \
                self.sample_data[idx_rand, indices[idx_sample, curr_feature_pos:]]

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

    def explain_text(self, text_data: Union[str, Tuple[str, ...]], label: Optional[int] = 0,
                     min_samples_per_feature: Optional[int] = 100, max_samples: Optional[int] = None):
        # Convert to representation of generator
        generator_instance = self.generator.to_internal([text_data], [label])

        # Generate new samples in representation of generator
        generated_samples = self.generator.generate(input_ids=generator_instance["input_ids"],
                                                    perturbable_mask=generator_instance["perturbable_mask"],
                                                    num_samples=self.num_generated_samples,
                                                    label=label,
                                                    **generator_instance["aux_data"])

        # Convert from representation of generator to text
        generated_text = self.generator.from_internal(generated_samples)

        # Convert from text to representation of interpreted model
        sample_data = self.model.to_internal(generated_text)
        self.update_sample_data(sample_data["input_ids"])

        # Convert instance being interpreted to representation of interpreted model
        model_instance = self.model.to_internal([text_data])
        input_ids = model_instance["input_ids"]
        perturbable_mask = model_instance["perturbable_mask"]

        # Note down the indices of examples which have a certain feature different from instance
        all_indices = torch.arange(self.sample_data.shape[0])
        for idx_feature in range(input_ids.shape[1]):
            if not perturbable_mask[0, idx_feature]:
                self.valid_indices.append([])
                continue

            different_example_inds = all_indices[self.sample_data[:, idx_feature] != input_ids[0, idx_feature]]
            if different_example_inds.shape[0] == 0:
                print(f"Warning: No unique values were found in generated data for feature {idx_feature}")
                different_example_inds = torch.arange(self.sample_data.shape[0])

            self.valid_indices.append(different_example_inds.tolist())

        res = super().explain(input_ids, label, perturbable_mask=perturbable_mask,
                              min_samples_per_feature=min_samples_per_feature, max_samples=max_samples,
                              **model_instance["aux_data"])
        res["input"] = self.model.convert_ids_to_tokens(model_instance["input_ids"])[0]

        return res


if __name__ == "__main__":
    model = InterpretableBertForSequenceClassification(tokenizer_name="bert-base-uncased",
                                                       model_name="/home/matej/Documents/embeddia/interpretability/ime-lm/examples/weights/snli_bert_uncased",
                                                       batch_size=2,
                                                       device="cpu")
    generator = BertForMaskedLMGenerator(tokenizer_name="bert-base-uncased",
                                         model_name="bert-base-uncased",
                                         batch_size=2,
                                         device="cpu")

    explainer = IMEMaskedLMExplainer(model=model,
                                     generator=generator,
                                     return_samples=True,
                                     return_scores=True,
                                     return_variance=True,
                                     return_num_samples=True,
                                     num_generated_samples=4)

    seq = ("A patient is being worked on by doctors and nurses", "A man is sleeping.")
    res = explainer.explain_text(seq, label=2, min_samples_per_feature=10)
    for curr_token, curr_imp in zip(res["input"], res["importance"]):
        print(f"{curr_token} = {curr_imp: .4f}")
