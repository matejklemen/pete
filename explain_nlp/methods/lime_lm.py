from typing import Optional, List, Callable

import torch

from explain_nlp.generation.decoding import filter_factory
from explain_nlp.generation.generation_base import SampleGenerator
from explain_nlp.methods.lime import LIMEExplainer
from explain_nlp.modeling.modeling_base import InterpretableModel
from explain_nlp.utils import EncodingException


class LIMEMaskedLMExplainer(LIMEExplainer):
    def __init__(self, model: InterpretableModel, generator: SampleGenerator, kernel_width=1.0,
                 return_samples: Optional[bool] = False, return_scores: Optional[bool] = False,
                 return_metrics: Optional[bool] = True, shared_vocabulary: Optional[bool] = False,
                 label_func: Optional[Callable[[int, int], List[int]]] = None):
        super().__init__(model=model, kernel_width=kernel_width, return_samples=return_samples,
                         return_scores=return_scores, return_metrics=return_metrics,
                         shared_vocabulary=shared_vocabulary)
        self.generator = generator
        # In order for the 0/1 simplified representation to make sense, 0 means that the word is different
        self.generator.filters = [filter_factory("unique")] + self.generator.filters

        # Takes in (curr_label, num_samples), returns selected_labels
        self.label_func = label_func
        self.has_custom_label_func = self.label_func is not None

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

    def generate_neighbourhood(self, samples: torch.Tensor, removal_mask, label: Optional[int] = 0,
                               **generation_kwargs):
        num_samples = removal_mask.shape[0]
        if hasattr(self.generator, "label_weights"):
            if self.has_custom_label_func:
                randomly_selected_label = self.label_func(label, num_samples)
            else:
                randomly_selected_label = torch.multinomial(self.generator.label_weights,
                                                            num_samples=num_samples, replacement=True).tolist()
            randomly_selected_label = [self.generator.control_labels_str[i] for i in randomly_selected_label]
        else:
            randomly_selected_label = [None] * num_samples

        generated_examples = self.generator.generate_masked_samples(samples,
                                                                    generation_mask=removal_mask,
                                                                    control_labels=randomly_selected_label,
                                                                    **generation_kwargs)
        if self.shared_vocabulary:
            model_examples = {
                "input_ids": generated_examples,
                "aux_data": generation_kwargs
            }
        else:
            text_examples = self.generator.from_internal(generated_examples, **generation_kwargs)
            model_examples = self.model.to_internal(text_examples)

        return model_examples["input_ids"]


if __name__ == "__main__":
    from explain_nlp.modeling.modeling_transformers import InterpretableBertForSequenceClassification
    from explain_nlp.generation.generation_transformers import BertForControlledMaskedLMGenerator
    from explain_nlp.visualizations.highlight import highlight_plot
    from explain_nlp.visualizations.internal import visualize_lime_internals

    model = InterpretableBertForSequenceClassification(
        model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/classifiers/snli_bert_uncased",
        tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/classifiers/snli_bert_uncased",
        batch_size=8,
        max_seq_len=41,
        device="cpu"
    )

    generator = BertForControlledMaskedLMGenerator(
        model_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/language_models/bert_snli_clm_best",
        tokenizer_name="/home/matej/Documents/embeddia/interpretability/explain_nlp/resources/weights/language_models/bert_snli_clm_best",
        batch_size=8,
        max_seq_len=42,
        device="cpu",
        strategy="top_p",
        top_p=0.01,
        control_labels=["<ENTAILMENT>", "<NEUTRAL>", "<CONTRADICTION>"]
    )

    explainer = LIMEMaskedLMExplainer(model, generator=generator, return_samples=True, return_scores=True,
                                      kernel_width=1.0, shared_vocabulary=True)

    seq = ("A shirtless man skateboards on a ledge.", "A man without a shirt.")
    EXPLAINED_LABEL = 0
    EXPLANATION_LENGTH = 5
    res = explainer.explain_text(seq, label=EXPLAINED_LABEL, num_samples=10, explanation_length=EXPLANATION_LENGTH)

    visualize_lime_internals(sequence_tokens=res["input"][:19],
                             token_mask=[res["indicators"][_i][:19] for _i in range(len(res["indicators"]))],
                             probabilities=res["scores"],
                             width_per_sample=0.1,
                             height_per_token=0.2,
                             ylabel="Probability (entailment)",
                             sort_key="token_mask")
    highlight_plot([res["input"]],
                   importances=[res["importance"].tolist()],
                   pred_labels=["entailment"],
                   actual_labels=["entailment"],
                   path="tmp_lime.html")

