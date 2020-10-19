from typing import Optional

import torch
from transformers import BertTokenizer, BertForMaskedLM

from explain_nlp.methods.ime import IMEExplainer

# in BERT pretraining, 15% of the tokens are masked - increasing this number decreases the available context and
# likely the quality of generated sequences
from explain_nlp.methods.modeling import InterpretableModel, InterpretableBertForSequenceClassification

MLM_MASK_PROPORTION = 0.15

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class IMEMaskedLMExplainer(IMEExplainer):
    def __init__(self, model: InterpretableModel, pretrained_name_or_path: Optional[str] = "bert-base-uncased",
                 batch_size: Optional[int] = 2, top_p: Optional[float] = 0.9, confidence_interval: Optional[int] = None,
                 max_abs_error: Optional[int] = None, num_generated_samples: Optional[int] = 10,
                 return_variance: Optional[bool] = False, return_num_samples: Optional[bool] = False,
                 return_samples: Optional[bool] = False, return_scores: Optional[bool] = False):
        # IME requires sampling data so we give it dummy data and later override it with generated data
        dummy_sample_data = torch.empty((0, 0), dtype=torch.long)
        super().__init__(sample_data=dummy_sample_data, model=model, confidence_interval=confidence_interval,
                         max_abs_error=max_abs_error, return_variance=return_variance,
                         return_num_samples=return_num_samples, return_samples=return_samples,
                         return_scores=return_scores)

        self.num_generated_samples = num_generated_samples
        self.mlm_batch_size = batch_size
        self.top_p = top_p

        self.mlm_generator = BertForMaskedLM.from_pretrained(pretrained_name_or_path).to(DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_name_or_path)
        self.mlm_generator.eval()

    def generate_samples(self, instance: torch.Tensor, perturbable_mask: torch.Tensor, **kwargs):
        num_features = int(instance.shape[1])
        perturbable_inds = torch.arange(num_features)[perturbable_mask[0]]
        num_perturbable = perturbable_inds.shape[0]
        masked_samples = instance.repeat((self.num_generated_samples, 1))

        # Mask and predict all tokens, one token at a time, in different order - slightly diverse greedy decoding
        probas = torch.zeros((self.num_generated_samples, num_features))
        probas[:, perturbable_inds] = 1 / num_perturbable
        indices = torch.multinomial(probas, num_samples=num_perturbable)

        token_type_ids = kwargs.get("token_type_ids")
        attention_mask = kwargs.get("attention_mask")

        for i in range(num_perturbable):
            curr_masked = indices[:, i]
            masked_samples[torch.arange(self.num_generated_samples), curr_masked] = self.tokenizer.mask_token_id

            aux_data = {
                "token_type_ids": token_type_ids.repeat((self.mlm_batch_size, 1)).to(DEVICE),
                "attention_mask": attention_mask.repeat((self.mlm_batch_size, 1)).to(DEVICE)
            }

            num_batches = (self.num_generated_samples + self.mlm_batch_size - 1) // self.mlm_batch_size
            for idx_batch in range(num_batches):
                s_batch, e_batch = idx_batch * self.mlm_batch_size, (idx_batch + 1) * self.mlm_batch_size
                curr_input_ids = masked_samples[s_batch: e_batch]
                curr_batch_size = curr_input_ids.shape[0]

                generator_data = {
                    "input_ids": curr_input_ids.to(DEVICE),
                    "token_type_ids": aux_data["token_type_ids"][: curr_batch_size],
                    "attention_mask": aux_data["attention_mask"][: curr_batch_size]
                }

                res = self.mlm_generator(**generator_data, return_dict=True)
                logits = res["logits"][torch.arange(curr_batch_size), curr_masked[s_batch: e_batch]]
                greedy_preds = torch.argmax(logits, dim=-1, keepdim=True)  # shape: [curr_batch_size, 1]

                masked_samples[torch.arange(s_batch, s_batch + curr_batch_size),
                               curr_masked[s_batch: e_batch]] = greedy_preds[:, 0].cpu()

        self.update_sample_data(masked_samples)

    def explain(self, instance: torch.Tensor, label: Optional[int] = 0, perturbable_mask: Optional[torch.Tensor] = None,
                min_samples_per_feature: Optional[int] = 100, max_samples: Optional[int] = None,
                **modeling_kwargs):
        eff_perturbable_mask = perturbable_mask if perturbable_mask is not None \
            else torch.ones((1, instance.shape[1]), dtype=torch.bool)
        self.generate_samples(instance, perturbable_mask=eff_perturbable_mask, **modeling_kwargs)
        return super().explain(instance, label, perturbable_mask=eff_perturbable_mask,
                               min_samples_per_feature=min_samples_per_feature, max_samples=max_samples,
                               **modeling_kwargs)


if __name__ == "__main__":
    model = InterpretableBertForSequenceClassification(tokenizer_name="bert-base-uncased",
                                                       model_name="/home/matej/Documents/embeddia/interpretability/ime-lm/examples/weights/snli_bert_uncased",
                                                       batch_size=4,
                                                       device="cpu")
    explainer = IMEMaskedLMExplainer(model=model,
                                     pretrained_name_or_path="bert-base-uncased",
                                     return_samples=True,
                                     return_scores=True,
                                     return_variance=True,
                                     return_num_samples=True,
                                     num_generated_samples=10,
                                     batch_size=4)

    example = model.to_internal(("My name is Iron Man", "I am Iron Man"))
    res = explainer.explain(example["input_ids"],
                            perturbable_mask=example["perturbable_mask"],
                            min_samples_per_feature=3,
                            token_type_ids=example["token_type_ids"], attention_mask=example["attention_mask"])
    print(res["importance"])
