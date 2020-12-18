from typing import List, Dict, Union, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification


# TODO: mask, mask_token_id properties
# TODO: mask token in generator is not necessarily same as mask token in model (e.g. <MASK> vs [MASK])
class InterpretableModel:
    def from_internal(self, encoded_data: torch.Tensor,
                      skip_special_tokens: bool = True,
                      take_as_single_sequence: bool = False) -> List[Union[str, Tuple[str, ...]]]:
        """ Convert from internal model representation to text."""
        raise NotImplementedError

    def to_internal(self, text_data: List[Union[str, Tuple[str, ...]]]) -> Dict:
        """ Convert from text to internal model representation. Make sure to include 'perturbable_mask' in the
        returned dictionary."""
        raise NotImplementedError

    def convert_ids_to_tokens(self, ids: torch.Tensor) -> List[List[str]]:
        """ Convert integer-encoded tokens to str-encoded tokens, but keep them split."""
        raise NotImplementedError

    def score(self, input_ids: torch.Tensor, **aux_data):
        """ Obtain scores (e.g. probabilities for classes) for encoded data. Make sure to handle batching here.

        `aux_data` will contain all the auxiliary data required to do the modeling: one batch-first instance of the
        data, which will be repeated along first axis to match dimension of a batch of `input_ids`.
        """
        raise NotImplementedError

    @property
    def special_token_ids(self):
        raise NotImplementedError


class InterpretableBertForSequenceClassification(InterpretableModel):
    def __init__(self, tokenizer_name, model_name, batch_size=8, max_seq_len=64, max_words: Optional[int] = 16,
                 device="cuda"):
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_words = max_words

        assert device in ["cpu", "cuda"]
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Device is set to 'cuda', but no CUDA device could be found. If you want to run the model "
                             "on CPU, set device to 'cpu'")
        self.device = torch.device(device)

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @property
    def special_token_ids(self):
        return self.tokenizer.all_special_ids

    def from_internal(self, encoded_data, skip_special_tokens: bool = True, take_as_single_sequence: bool = False):
        decoded_data = []
        for idx_example in range(encoded_data.shape[0]):
            sep_tokens = torch.nonzero(encoded_data[idx_example] == self.tokenizer.sep_token_id, as_tuple=False)

            # Multiple sequences present: [CLS] <seq1> [SEP] <seq2> [SEP] -> (<seq1>, <seq2>)
            if sep_tokens.shape[0] > 1 and not take_as_single_sequence:
                bnd = int(sep_tokens[0])
                seq1 = self.tokenizer.decode(encoded_data[idx_example, :bnd], skip_special_tokens=skip_special_tokens)
                seq2 = self.tokenizer.decode(encoded_data[idx_example, bnd + 1:], skip_special_tokens=skip_special_tokens)
                decoded_data.append((seq1, seq2))
            else:
                decoded_data.append(self.tokenizer.decode(encoded_data[idx_example], skip_special_tokens=skip_special_tokens))

        return decoded_data

    def to_internal(self, text_data):
        res = self.tokenizer.batch_encode_plus(text_data, return_special_tokens_mask=True, return_tensors="pt",
                                               padding="max_length", max_length=self.max_seq_len,
                                               truncation="longest_first")
        formatted_res = {
            "input_ids": res["input_ids"],
            "perturbable_mask": torch.logical_not(res["special_tokens_mask"]),
            "aux_data": {
                "token_type_ids": res["token_type_ids"],
                "attention_mask": res["attention_mask"]
            }
        }

        return formatted_res

    def words_to_internal(self, text_data: List[Union[List[str], Tuple[List[str], ...]]]) -> Dict:
        """ Convert examples into model's representation, keeping the word boundaries intact.

        Args:
        -----
        text_data:
            Pre-tokenized examples or example pairs
        """
        is_seq_pair = isinstance(text_data[0], tuple)
        ret_dict = {
            "words": [],  # text data, augmented with any additional control tokens (used to align importances)
            "input_ids": [],  # list of encoded token subwords for each example/example pair; type: List[List[List[int]]
            # word-level annotations
            "perturbable_mask": [],
            "aux_data": {"token_type_ids": [], "attention_mask": []}
        }

        for curr_example in text_data:
            raw_example = []
            formatted_example = []
            ex0, ex1 = curr_example if is_seq_pair else (curr_example, None)
            formatted_ex0, formatted_ex1 = [], None

            for curr_token in ex0:
                formatted_ex0.append(self.tokenizer.encode(curr_token, add_special_tokens=False))
            proxy_ex0 = ["a"] * len(formatted_ex0)

            if is_seq_pair:
                formatted_ex1 = []
                for curr_token in ex1:
                    formatted_ex1.append(self.tokenizer.encode(curr_token, add_special_tokens=False))

            proxy_ex1 = ["b"] * len(formatted_ex1) if is_seq_pair else None
            curr_seq_len = len(proxy_ex0) + (len(proxy_ex1) if is_seq_pair else 0)
            num_special_tokens = 3 if is_seq_pair else 2

            res = self.tokenizer.encode_plus(text=proxy_ex0, text_pair=proxy_ex1,
                                             return_special_tokens_mask=True, return_tensors="pt",
                                             padding="max_length", max_length=self.max_words,
                                             truncation="longest_first")
            proxy_ex0, proxy_ex1, _ = self.tokenizer.truncate_sequences(ids=proxy_ex0, pair_ids=proxy_ex1,
                                                                        num_tokens_to_remove=((curr_seq_len + num_special_tokens) - self.max_words))
            formatted_ex0 = formatted_ex0[: len(proxy_ex0)]
            formatted_ex1 = formatted_ex1[: len(proxy_ex1)]
            ex0 = ex0[: len(proxy_ex0)]
            ex1 = ex1[: len(proxy_ex1)]

            formatted_example.append([self.tokenizer.cls_token_id])
            formatted_example.extend(formatted_ex0)
            formatted_example.append([self.tokenizer.sep_token_id])
            raw_example.append(self.tokenizer.cls_token)
            raw_example.extend(ex0)
            raw_example.append(self.tokenizer.sep_token)
            if is_seq_pair:
                formatted_example.extend(formatted_ex1)
                formatted_example.append([self.tokenizer.sep_token_id])
                raw_example.extend(ex1)
                raw_example.append(self.tokenizer.sep_token)

            formatted_example.extend([[self.tokenizer.pad_token_id]
                                      for _ in range(self.max_words - len(formatted_example))])
            raw_example.extend([self.tokenizer.pad_token for _ in range(self.max_words - len(formatted_example))])

            ret_dict["words"].append(raw_example)
            ret_dict["input_ids"].append(formatted_example)
            ret_dict["perturbable_mask"].append(torch.logical_not(res["special_tokens_mask"]))
            ret_dict["aux_data"]["token_type_ids"].append(res["token_type_ids"])
            ret_dict["aux_data"]["attention_mask"].append(res["attention_mask"])

        ret_dict["perturbable_mask"] = torch.cat(ret_dict["perturbable_mask"])
        ret_dict["aux_data"]["token_type_ids"] = torch.cat(ret_dict["aux_data"]["token_type_ids"])
        ret_dict["aux_data"]["attention_mask"] = torch.cat(ret_dict["aux_data"]["attention_mask"])

        return ret_dict

    def convert_ids_to_tokens(self, ids):
        return [self.tokenizer.convert_ids_to_tokens(curr_ids) for curr_ids in ids.tolist()]

    @torch.no_grad()
    def score(self, input_ids: torch.Tensor, **kwargs):
        # If a single example is provided, broadcast it to input size, otherwise assume correct shapes are provided
        if kwargs["token_type_ids"].shape[0] != input_ids.shape[0]:
            aux_data = {additional_arg: kwargs[additional_arg].repeat((input_ids.shape[0], 1))
                        for additional_arg in ["token_type_ids", "attention_mask"]}
        else:
            aux_data = {additional_arg: kwargs[additional_arg]
                        for additional_arg in ["token_type_ids", "attention_mask"]}

        num_total_batches = (input_ids.shape[0] + self.batch_size - 1) // self.batch_size
        probas = torch.zeros((input_ids.shape[0], self.model.config.num_labels))
        for idx_batch in range(num_total_batches):
            s_b, e_b = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
            curr_input_ids = input_ids[s_b: e_b].to(self.device)
            res = self.model(curr_input_ids, **{k: v[s_b: e_b].to(self.device) for k, v in aux_data.items()},
                             return_dict=True)

            probas[s_b: e_b] = F.softmax(res["logits"], dim=-1)

        return probas


class DummySentiment(InterpretableModel):
    """ Dummy 2-word sentiment prediction (positive/negative). """
    def __init__(self):
        vocab = ["allegedly", "achingly", "amazingly", "astonishingly", "not", "very", "surprisingly", "good", "bad"]
        self.tok2id = {"<UNK>": 0, "<PAD>": 1}
        self.id2tok = {0: "<UNK>", 1: "<PAD>"}

        for i, word in enumerate(vocab, start=2):
            self.tok2id[word] = i
            self.id2tok[i] = word

        self.model_scores = {
            self.tok2id["allegedly"]: {
                self.tok2id["bad"]: [0.5, 0.5],
                self.tok2id["good"]: [0.5, 0.5],
                self.tok2id["<PAD>"]: [0.5, 0.5],
                self.tok2id["<UNK>"]: [0.5, 0.5]
            },
            self.tok2id["achingly"]: {
                self.tok2id["bad"]: [0.55, 0.45],
                self.tok2id["good"]: [0.45, 0.55],
                self.tok2id["<PAD>"]: [0.52, 0.48],
                self.tok2id["<UNK>"]: [0.52, 0.48]
            },
            self.tok2id["amazingly"]: {
                self.tok2id["bad"]: [0.8, 0.2],
                self.tok2id["good"]: [0.2, 0.8],
                self.tok2id["<PAD>"]: [0.45, 0.55],
                self.tok2id["<UNK>"]: [0.45, 0.55]
            },
            self.tok2id["astonishingly"]: {
                self.tok2id["bad"]: [0.9, 0.1],
                self.tok2id["good"]: [0.1, 0.9],
                self.tok2id["<PAD>"]: [0.5, 0.5],
                self.tok2id["<UNK>"]: [0.5, 0.5]
            },
            self.tok2id["not"]: {
                self.tok2id["bad"]: [0.35, 0.65],
                self.tok2id["good"]: [0.65, 0.35],
                self.tok2id["<PAD>"]: [0.5, 0.5],
                self.tok2id["<UNK>"]: [0.5, 0.5]
            },
            self.tok2id["very"]: {
                self.tok2id["bad"]: [1.0, 0.0],
                self.tok2id["good"]: [0.0, 1.0],
                self.tok2id["<PAD>"]: [0.5, 0.5],
                self.tok2id["<UNK>"]: [0.5, 0.5]
            },
            self.tok2id["surprisingly"]: {
                self.tok2id["bad"]: [0.55, 0.45],
                self.tok2id["good"]: [0.45, 0.55],
                self.tok2id["<PAD>"]: [0.5, 0.5],
                self.tok2id["<UNK>"]: [0.5, 0.5]
            },
            self.tok2id["<PAD>"]: {
                self.tok2id["bad"]: [0.7, 0.3],
                self.tok2id["good"]: [0.3, 0.7],
                self.tok2id["<PAD>"]: [0.5, 0.5],
                self.tok2id["<UNK>"]: [0.5, 0.5]
            },
            self.tok2id["<UNK>"]: {
                self.tok2id["bad"]: [0.7, 0.3],
                self.tok2id["good"]: [0.3, 0.7],
                self.tok2id["<PAD>"]: [0.5, 0.5],
                self.tok2id["<UNK>"]: [0.5, 0.5]
            }
        }

    @property
    def special_token_ids(self):
        return [self.tok2id["<PAD>"], self.tok2id["<UNK>"]]

    def from_internal(self, encoded_data: torch.Tensor, skip_special_tokens: bool = True,
                      take_as_single_sequence: bool = False) -> List[str]:
        return [" ".join([self.id2tok[i] for i in sequence]) for sequence in encoded_data.tolist()]

    def to_internal(self, text_data: List[str]) -> Dict:
        tokenized_examples = [text.split(" ") for text in text_data]
        encoded_tokens = []
        # Encode and pad/truncate to max length
        for example_tokens in tokenized_examples:
            curr_encoded = [self.tok2id.get(t.lower(), self.tok2id["<UNK>"]) for t in example_tokens]
            encoded_tokens.append(curr_encoded)

        return {
            "input_ids": torch.tensor(encoded_tokens)
        }

    def convert_ids_to_tokens(self, ids):
        str_tokens = []
        for curr_ids in ids.tolist():
            str_tokens.append([self.id2tok[i] for i in curr_ids])

        return str_tokens

    def score(self, input_ids: torch.Tensor, **aux_data):
        scores = []
        for i in range(input_ids.shape[0]):
            scores.append(self.model_scores[int(input_ids[i, 0])][int(input_ids[i, 1])])

        return torch.tensor(scores, dtype=torch.float32)


if __name__ == "__main__":
    model = InterpretableBertForSequenceClassification(tokenizer_name="bert-base-uncased",
                                                       model_name="/home/matej/Documents/embeddia/interpretability/ime-lm/examples/weights/snli_bert_uncased",
                                                       batch_size=4,
                                                       device="cpu")

    encoded = model.to_internal([("I am Iron Man", "My name is Iron Man"), ("Do not blink", "Blink and you're dead")])
    print(encoded)
    print(model.from_internal(encoded["input_ids"]))
    probas = model.score(encoded["input_ids"], **encoded["aux_data"])
    print(probas)
