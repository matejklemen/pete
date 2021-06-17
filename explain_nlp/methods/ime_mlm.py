from typing import Optional, Union, Tuple, List

import torch

from explain_nlp.generation.generation_base import SampleGenerator
from explain_nlp.methods.ime import IMEExplainer
from explain_nlp.modeling.modeling_base import InterpretableModel


class IMEExternalLMExplainer(IMEExplainer):
    def __init__(self, model: InterpretableModel, generator: SampleGenerator,
                 confidence_interval: Optional[float] = None,  max_abs_error: Optional[float] = None,
                 num_generated_samples: Optional[int] = 10, return_num_samples: Optional[bool] = False,
                 return_samples: Optional[bool] = False, return_scores: Optional[bool] = False,
                 criterion: Optional[str] = "squared_error"):
        # IME requires sampling data so we give it dummy data and later override it with generated data
        dummy_sample_data = torch.randint(5, (1, 1), dtype=torch.long)
        super().__init__(sample_data=dummy_sample_data, model=model, confidence_interval=confidence_interval,
                         max_abs_error=max_abs_error, return_num_samples=return_num_samples,
                         return_samples=return_samples, return_scores=return_scores, criterion=criterion,
                         shared_vocabulary=True)  # fixed to True because we want to use base IME's logic

        self.generator = generator
        self.num_generated_samples = num_generated_samples

    def explain_text(self, text_data: Union[str, Tuple[str, ...]], label: Optional[int] = 0,
                     min_samples_per_feature: Optional[int] = 100, max_samples: Optional[int] = None,
                     exact_samples_per_feature: Optional[torch.Tensor] = None,
                     pretokenized_text_data: Optional[Union[List[str], Tuple[List[str], ...]]] = None,
                     custom_features: Optional[List[List[int]]] = None):
        # Convert to representation of generator
        is_split_into_units = pretokenized_text_data is not None
        generator_instance = self.generator.to_internal([pretokenized_text_data if is_split_into_units else text_data],
                                                        is_split_into_units=is_split_into_units)

        # Generate new samples in representation of generator
        generator_res = self.generator.generate(input_ids=generator_instance["input_ids"],
                                                perturbable_mask=generator_instance["perturbable_mask"],
                                                num_samples=self.num_generated_samples,
                                                label=label,
                                                **generator_instance["aux_data"])

        # Convert from representation of generator to text
        generated_text = self.generator.from_internal(generator_res["input_ids"],
                                                      **generator_instance["aux_data"])

        for i in range(len(generated_text)):
            print(generated_text[i])

        # Convert from text to representation of interpreted model
        sample_data = self.model.to_internal(generated_text)
        self.update_sample_data(sample_data["input_ids"])

        return super().explain_text(text_data=text_data, label=label,
                                    min_samples_per_feature=min_samples_per_feature, max_samples=max_samples,
                                    exact_samples_per_feature=exact_samples_per_feature,
                                    pretokenized_text_data=pretokenized_text_data,
                                    custom_features=custom_features)


if __name__ == "__main__":
    from explain_nlp.generation.generation_transformers import BertForControlledMaskedLMGenerator
    from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification

    model = InterpretableBertForSequenceClassification(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased",
                                                       model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/snli_bert_uncased",
                                                       batch_size=10,
                                                       max_seq_len=41,
                                                       device="cpu")
    # LANG_MODEL_HANDLE = "bert-base-uncased"
    generator = BertForControlledMaskedLMGenerator(tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert_snli_clm_best",
                                                   model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/bert_snli_clm_best",
                                                   max_seq_len=41,
                                                   control_labels=["<ENTAILMENT>", "<NEUTRAL>", "<CONTRADICTION>"],
                                                   batch_size=10,
                                                   strategy="top_k",
                                                   top_k=3,
                                                   device="cpu")

    explainer = IMEExternalLMExplainer(model=model,
                                       generator=generator,
                                       return_samples=True,
                                       return_scores=True,
                                       return_num_samples=True,
                                       num_generated_samples=10)

    seq = ("A patient is being worked on by doctors and nurses", "A man is sleeping.")
    res = explainer.explain_text(seq, label=2, min_samples_per_feature=10)
    for curr_token, curr_imp in zip(res["input"], res["importance"]):
        print(f"{curr_token} = {curr_imp: .4f}")
