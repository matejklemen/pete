from typing import Optional, Union, Tuple, List

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
        dummy_sample_data = torch.randint(5, (1, 1), dtype=torch.long)
        super().__init__(sample_data=dummy_sample_data, model=model, confidence_interval=confidence_interval,
                         max_abs_error=max_abs_error, return_variance=return_variance,
                         return_num_samples=return_num_samples, return_samples=return_samples,
                         return_scores=return_scores)

        self.generator = generator
        self.num_generated_samples = num_generated_samples

    def explain_text(self, text_data: Union[str, Tuple[str, ...]], label: Optional[int] = 0,
                     min_samples_per_feature: Optional[int] = 100, max_samples: Optional[int] = None):
        # Convert to representation of generator
        generator_instance = self.generator.to_internal([text_data], [label])

        # Generate new samples in representation of generator
        # TODO: figure out how to make weights doable when using different model and generator
        #  (tokens can be misaligned, split differently, etc.) -- currently assuming SAME model and generator tokenization
        generator_res = self.generator.generate(input_ids=generator_instance["input_ids"],
                                                perturbable_mask=generator_instance["perturbable_mask"],
                                                num_samples=self.num_generated_samples,
                                                label=label,
                                                **generator_instance["aux_data"])
        generated_samples = generator_res["input_ids"]
        weights = generator_res["weights"]

        # Convert from representation of generator to text
        generated_text = self.generator.from_internal(generated_samples)

        # Convert from text to representation of interpreted model
        sample_data = self.model.to_internal(generated_text)

        # Convert instance being interpreted to representation of interpreted model
        model_instance = self.model.to_internal([text_data])
        input_ids = model_instance["input_ids"]
        perturbable_mask = model_instance["perturbable_mask"]

        # Note down the indices of examples which have a certain feature different from instance
        all_indices = torch.arange(sample_data["input_ids"].shape[0])
        for idx_feature in range(input_ids.shape[1]):
            if not perturbable_mask[0, idx_feature]:
                weights[:, idx_feature] = 0.0
                continue

            different_example_mask = sample_data["input_ids"][:, idx_feature] != input_ids[0, idx_feature]
            different_example_inds = all_indices[different_example_mask]
            if different_example_inds.shape[0] == 0:
                print(f"Warning: No unique values were found in generated data for feature {idx_feature}")
            else:
                weights[torch.logical_not(different_example_mask), idx_feature] = 0.0

        self.update_sample_data(sample_data["input_ids"], weights)

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
                                         strategy="num_samples",
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
