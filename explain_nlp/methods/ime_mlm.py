from typing import Optional, Union, Tuple

import torch

from explain_nlp.methods.generation import SampleGenerator, BertForMaskedLMGenerator
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.methods.modeling import InterpretableModel, InterpretableBertForSequenceClassification

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

    def explain_text(self, text_data: Union[str, Tuple[str, ...]], label: Optional[int] = 0,
                     min_samples_per_feature: Optional[int] = 100, max_samples: Optional[int] = None):
        # Convert to representation of generator
        generator_instance = self.generator.to_internal([text_data])

        # Generate new samples in representation of generator
        generated_samples = self.generator.generate(input_ids=generator_instance["input_ids"],
                                                    perturbable_mask=generator_instance["perturbable_mask"],
                                                    num_samples=self.num_generated_samples,
                                                    **generator_instance["aux_data"])

        # Convert from representation of generator to text
        generated_text = self.generator.from_internal(generated_samples)

        # Convert from text to representation of interpreted model
        sample_data = self.model.to_internal(generated_text)
        self.update_sample_data(sample_data["input_ids"])

        # Convert instance being interpreted to representation of interpreted model
        model_instance = self.model.to_internal([text_data])

        res = super().explain(model_instance["input_ids"], label, perturbable_mask=model_instance["perturbable_mask"],
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
