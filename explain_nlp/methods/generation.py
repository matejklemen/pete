from typing import Tuple, Union, Dict, List, Optional
from warnings import warn

import torch
from transformers import BertTokenizer, BertForMaskedLM

from explain_nlp.methods.decoding import greedy_decoding, top_p_decoding


class SampleGenerator:
    def from_internal(self, encoded_data: torch.Tensor) -> List[Union[str, Tuple[str, ...]]]:
        """ Convert from internal generator representation to text."""
        raise NotImplementedError

    def to_internal(self, text_data: List[Union[str, Tuple[str, ...]]]) -> Dict:
        """ Convert from text to internal generator representation.
        Make sure to include 'input_ids', 'perturbable_mask' and 'aux_data' in the returned dictionary."""
        raise NotImplementedError

    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor, num_samples: int,
                 **aux_data) -> torch.Tensor:
        raise NotImplementedError


class BertForMaskedLMGenerator(SampleGenerator):
    MLM_MAX_MASK_PROPORTION = 0.15

    def __init__(self, tokenizer_name, model_name, batch_size=8, device="cuda", top_p: Optional[float] = None,
                 masked_at_once: Optional[Union[int, float]] = 1, p_ensure_different: Optional[float] = 0.0):
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.batch_size = batch_size

        self.top_p = top_p
        self.masked_at_once = masked_at_once
        self.p_ensure_different = p_ensure_different

        assert device in ["cpu", "cuda"]
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Device is set to 'cuda', but no CUDA device could be found. If you want to run the model "
                             "on CPU, set device to 'cpu'")
        self.device = torch.device(device)

        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        self.generator = BertForMaskedLM.from_pretrained(self.model_name).to(self.device)
        self.generator.eval()

    def from_internal(self, encoded_data):
        decoded_data = []
        for idx_example in range(encoded_data.shape[0]):
            sep_tokens = torch.nonzero(encoded_data[idx_example] == self.tokenizer.sep_token_id, as_tuple=False)

            # Multiple sequences present: [CLS] <seq1> [SEP] <seq2> [SEP] -> (<seq1>, <seq2>)
            if sep_tokens.shape[0] == 2:
                bnd = int(sep_tokens[0])
                seq1 = self.tokenizer.decode(encoded_data[idx_example, :bnd], skip_special_tokens=True)
                seq2 = self.tokenizer.decode(encoded_data[idx_example, bnd + 1:], skip_special_tokens=True)
                decoded_data.append((seq1, seq2))
            else:
                decoded_data.append(self.tokenizer.decode(encoded_data[idx_example], skip_special_tokens=True))

        return decoded_data

    def to_internal(self, text_data):
        res = self.tokenizer.batch_encode_plus(text_data, return_special_tokens_mask=True, return_tensors="pt",
                                               padding="longest")
        formatted_res = {
            "input_ids": res["input_ids"],
            "perturbable_mask": torch.logical_not(res["special_tokens_mask"]),
            "aux_data": {
                "token_type_ids": res["token_type_ids"],
                "attention_mask": res["attention_mask"]
            }
        }

        return formatted_res

    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor, num_samples: int,
                 **aux_data) -> torch.Tensor:
        num_features = int(input_ids.shape[1])

        perturbable_inds = torch.arange(num_features)[perturbable_mask[0]]
        num_perturbable = perturbable_inds.shape[0]
        masked_samples = input_ids.repeat((num_samples, 1))

        probas = torch.zeros((num_samples, num_features))
        probas[:, perturbable_inds] = 1 / num_perturbable
        permuted_indices = torch.multinomial(probas, num_samples=num_perturbable)  # [num_samples, num_perturbable]

        token_type_ids = aux_data["token_type_ids"]
        attention_mask = aux_data["attention_mask"]

        generation_data = {
            "token_type_ids": token_type_ids.repeat((self.batch_size, 1)).to(self.device),
            "attention_mask": attention_mask.repeat((self.batch_size, 1)).to(self.device)
        }

        if self.top_p is None:
            def decoding_strategy(logits, ensure_diff_from):
                return greedy_decoding(logits, ensure_diff_from)
        else:
            def decoding_strategy(logits, ensure_diff_from):
                return top_p_decoding(logits, self.top_p, ensure_diff_from)

        mask_size = self.masked_at_once if isinstance(self.masked_at_once, int) \
            else int(self.masked_at_once * num_perturbable)
        mask_size = max(1, mask_size)

        max_recommended_mask_size = max(1, int(BertForMaskedLMGenerator.MLM_MAX_MASK_PROPORTION * num_perturbable))
        if mask_size > 1 and mask_size > max_recommended_mask_size:
            warn(f"More tokens are being masked than there usually were during BERT masked language modeling training. "
                 f"Recommended 'masked_at_once' size is <={max_recommended_mask_size} in this case")

        num_mask_chunks = (num_perturbable + mask_size - 1) // mask_size
        # Mask and predict all tokens, one token at a time, in different order - slightly diverse greedy decoding
        for idx_chunk in range(num_mask_chunks):
            curr_masked = permuted_indices[:, idx_chunk * mask_size: (idx_chunk + 1) * mask_size]
            curr_mask_size = curr_masked.shape[1]
            masked_samples[torch.arange(num_samples).unsqueeze(1), curr_masked] = self.tokenizer.mask_token_id

            num_batches = (num_samples + self.batch_size - 1) // self.batch_size
            for idx_batch in range(num_batches):
                s_batch, e_batch = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
                curr_input_ids = masked_samples[s_batch: e_batch]
                curr_batch_size = curr_input_ids.shape[0]

                res = self.generator(curr_input_ids.to(self.device),
                                     token_type_ids=generation_data["token_type_ids"][: curr_batch_size],
                                     attention_mask=generation_data["attention_mask"][: curr_batch_size],
                                     return_dict=True)
                logits = res["logits"][torch.arange(curr_batch_size).unsqueeze(1), curr_masked[s_batch: e_batch]]
                for idx_token in range(curr_mask_size):
                    ensure_different = torch.rand(()) < self.p_ensure_different

                    curr_token_logits = logits[:, idx_token]
                    curr_preds = decoding_strategy(curr_token_logits,
                                                   ensure_diff_from=input_ids[0, curr_masked[s_batch: e_batch]]
                                                   if ensure_different else None)

                    masked_samples[torch.arange(s_batch, s_batch + curr_batch_size),
                                   curr_masked[s_batch: e_batch, idx_token]] = curr_preds[:, 0].cpu()

        return masked_samples


if __name__ == "__main__":
    generator = BertForMaskedLMGenerator(tokenizer_name="bert-base-uncased",
                                         model_name="bert-base-uncased",
                                         batch_size=2,
                                         device="cpu")

    seq = ("My name is Iron Man", "I am Iron Man")
    encoded = generator.to_internal([seq])

    generator.generate(encoded["input_ids"], perturbable_mask=encoded["perturbable_mask"], num_samples=10,
                       **encoded["aux_data"])



