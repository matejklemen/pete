from typing import List, Dict, Union, Tuple

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification


class InterpretableModel:
    def from_internal(self, encoded_data: torch.Tensor) -> List[str]:
        """ Convert from internal model representation to text."""
        raise NotImplementedError

    def to_internal(self, text_data: Union[str, Tuple[str, ...]]) -> Dict:
        """ Convert from text to internal model representation. Make sure to include 'perturbable_mask' in the
        returned dictionary."""
        raise NotImplementedError

    def score(self, input_ids: torch.Tensor, **aux_data):
        """ Obtain scores (e.g. probabilities for classes) for encoded data. Make sure to handle batching here.

        `aux_data` should contain all the auxiliary data required to do the modeling: one batch-first instance of the
        data, which will be repeated along first axis to match dimension of a batch of `input_ids`.
        """
        raise NotImplementedError


class InterpretableBertForSequenceClassification(InterpretableModel):
    def __init__(self, tokenizer_name, model_name, batch_size=8, device="cuda"):
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.batch_size = batch_size

        assert device in ["cpu", "cuda"]
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Device is set to 'cuda', but no CUDA device could be found. If you want to run the model "
                             "on CPU, set device to 'cpu'")
        self.device = torch.device(device)

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def from_internal(self, encoded_data: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(encoded_data, skip_special_tokens=True)

    def to_internal(self, text_data: Union[str, Tuple[str, ...]]) -> Dict:
        res = self.tokenizer.encode_plus(*text_data if isinstance(text_data, tuple) else text_data,
                                         return_special_tokens_mask=True, return_tensors="pt")
        res["perturbable_mask"] = torch.logical_not(res["special_tokens_mask"])
        del res["special_tokens_mask"]

        return res

    @torch.no_grad()
    def score(self, input_ids: torch.Tensor, **kwargs):
        aux_data = {additional_arg: kwargs[additional_arg].repeat((self.batch_size, 1)).to(self.device)
                    for additional_arg in ["token_type_ids", "attention_mask"]}

        num_total_batches = (input_ids.shape[0] + self.batch_size - 1) // self.batch_size
        probas = torch.zeros((input_ids.shape[0], self.model.config.num_labels))
        for idx_batch in range(num_total_batches):
            s_b, e_b = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
            curr_input_ids = input_ids[s_b: e_b].to(self.device)
            curr_batch_size = curr_input_ids.shape[0]
            res = self.model(curr_input_ids, **{k: v[: curr_batch_size] for k, v in aux_data.items()},
                             return_dict=True)

            probas[s_b: e_b] = F.softmax(res["logits"], dim=-1)

        return probas


class InterpretableDummy(InterpretableModel):
    """ Dummy model example, only for debugging purpose. """
    def __init__(self):
        vocab = ["broccoli", "banana", "mug", "coffee", "paper", "cable", "bin"]
        self.tok2id = {"<UNK>": 0}
        self.id2tok = {0: "<UNK>"}

        for i, word in enumerate(vocab, start=1):
            self.tok2id[word] = i
            self.id2tok[i] = word

    def from_internal(self, encoded_data: torch.Tensor) -> List[str]:
        return [" ".join([self.id2tok[i] for i in sequence]) for sequence in encoded_data.tolist()]

    def to_internal(self, text_data: str) -> Dict:
        tokens = text_data.split(" ")
        return {
            "input_ids": torch.tensor([[self.tok2id.get(t, self.tok2id["<UNK>"]) for t in tokens]])
        }

    def score(self, input_ids: torch.Tensor, **aux_data):
        # [0] = number of non-handpicked words in sequence, [1] = number of handpicked words in sequence
        counts = [[0, 0] for _ in range(input_ids.shape[0])]

        CHOSEN = {self.tok2id["broccoli"], self.tok2id["banana"]}
        for i in range(input_ids.shape[0]):
            num_chosen = sum(w in CHOSEN for w in input_ids[i].tolist())
            counts[i][1] = num_chosen
            counts[i][0] = input_ids.shape[1] - num_chosen

        return F.softmax(torch.tensor(counts, dtype=torch.float32), dim=-1)


if __name__ == "__main__":
    model = InterpretableBertForSequenceClassification(tokenizer_name="bert-base-uncased",
                                                       model_name="/home/matej/Documents/embeddia/interpretability/ime-lm/examples/weights/snli_bert_uncased",
                                                       batch_size=4,
                                                       device="cpu")

    encoded = model.to_internal(("I am Iron Man", "My name is Iron Man"))
    del encoded["perturbable_mask"]
    probas = model.score(**encoded)
    print(probas)
