from typing import Optional, Union, Tuple, List

import torch

from explain_nlp.methods.generation import SampleGenerator
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.methods.modeling import InterpretableModel


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
                     min_samples_per_feature: Optional[int] = 100, max_samples: Optional[int] = None,
                     pretokenized_text_data: Optional[Union[List[str], Tuple[List[str], ...]]] = None,
                     custom_features: Optional[List[List[int]]] = None):
        # Convert to representation of generator
        generator_instance = self.generator.to_internal([text_data])

        # Generate new samples in representation of generator
        generator_res = self.generator.generate(input_ids=generator_instance["input_ids"],
                                                perturbable_mask=generator_instance["perturbable_mask"],
                                                num_samples=self.num_generated_samples,
                                                label=label,
                                                **generator_instance["aux_data"])
        generated_samples = generator_res["input_ids"]

        # Convert from representation of generator to text
        generated_text = self.generator.from_internal(generated_samples)

        for i in range(len(generated_text)):
            print(generated_text[i])

        # Convert from text to representation of interpreted model
        sample_data = self.model.to_internal(generated_text)
        self.update_sample_data(sample_data["input_ids"],
                                data_weights=None if "weights" not in generator_res else generator_res["weights"])

        # Find expectation of generated text
        generated_scores = self.model.score(sample_data["input_ids"], **sample_data["aux_data"])
        if "weights" in generator_res:
            expected_scores = torch.sum(generator_res["weights"].unsqueeze(1) * generated_scores, dim=0)
        else:
            expected_scores = torch.mean(generated_scores, dim=0)

        print(f"Expected scores of generated sample:")
        print(expected_scores)

        # Convert instance being interpreted to representation of interpreted model
        model_instance = self.model.to_internal([text_data],
                                                pretokenized_text_data=[pretokenized_text_data] if pretokenized_text_data is not None else None)
        input_ids = model_instance["input_ids"]
        perturbable_mask = model_instance["perturbable_mask"]

        res = super().explain(input_ids, label, perturbable_mask=perturbable_mask,
                              min_samples_per_feature=min_samples_per_feature, max_samples=max_samples,
                              custom_features=custom_features,
                              **model_instance["aux_data"])
        res["input"] = self.model.convert_ids_to_tokens(model_instance["input_ids"])[0]

        return res


if __name__ == "__main__":
    from explain_nlp.methods.generation import BertForControlledMaskedLMGenerator
    from explain_nlp.methods.modeling import InterpretableBertForSequenceClassification
    model = InterpretableBertForSequenceClassification(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased",
                                                       model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased",
                                                       batch_size=2,
                                                       device="cpu")
    LANG_MODEL_HANDLE = "/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert_snli_clm_best"
    # LANG_MODEL_HANDLE = "bert-base-uncased"
    generator = BertForControlledMaskedLMGenerator(tokenizer_name=LANG_MODEL_HANDLE,
                                                   model_name=LANG_MODEL_HANDLE,
                                                   control_labels=["<ENTAILMENT>", "<NEUTRAL>", "<CONTRADICTION>"],
                                                   batch_size=2,
                                                   strategy="greedy",
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
