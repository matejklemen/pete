import torch
from transformers import BertTokenizer, BertForMaskedLM

from explain_nlp.methods.ime import IMEExplainer

# in BERT pretraining, 15% of the tokens are masked - increasing this number decreases the available context and
# likely the quality of generated sequences
MLM_MASK_PROPORTION = 0.15

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class IMEMaskedLMExplainer(IMEExplainer):
    def __init__(self, model_func=None, pretrained_name_or_path="bert-base-uncased", batch_size=2, top_p=0.9,
                 return_variance=False, return_num_samples=False, return_samples=False, return_scores=False,
                 num_generated_samples=10):
        # IME requires sampling data so we give it dummy data and later override it with generated data
        dummy_sample_data = torch.empty((0, 0), dtype=torch.long)
        super().__init__(sample_data=dummy_sample_data, model_func=model_func,
                         return_variance=return_variance, return_num_samples=return_num_samples,
                         return_samples=return_samples, return_scores=return_scores)

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

    def explain(self, instance: torch.Tensor, label: int = 0, **kwargs):
        perturbable_mask = kwargs.get("perturbable_mask", torch.ones((1, instance.shape[1]), dtype=torch.bool))
        kwargs["perturbable_mask"] = perturbable_mask
        self.generate_samples(instance, **kwargs)
        return super().explain(instance, label, **kwargs)


if __name__ == "__main__":
    # dummy call example, only for debugging purpose
    def dummy_func(X):
        return torch.randn((X.shape[0], 2))

    explainer = IMEMaskedLMExplainer(model_func=dummy_func,
                                     pretrained_name_or_path="bert-base-uncased",
                                     return_samples=True,
                                     return_scores=True,
                                     return_variance=True,
                                     return_num_samples=True,
                                     num_generated_samples=100,
                                     batch_size=8)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    example = tokenizer.encode_plus("My name is Iron Man", "I am Iron Man", return_tensors="pt",
                                    return_special_tokens_mask=True, max_length=15, padding="max_length")
    res = explainer.explain(example["input_ids"],
                            perturbable_mask=torch.logical_not(example["special_tokens_mask"]),
                            min_samples_per_feature=10,
                            token_type_ids=example["token_type_ids"], attention_mask=example["attention_mask"])
    print(res["importance"])
