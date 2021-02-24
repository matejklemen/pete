from typing import List, Union, Tuple, Optional, Dict

import torch


# TODO: TokenizerBase in some utils package? tokenization methods are likely gonna be same/similar in model and generator
class InterpretableModel:
    @property
    def mask_token(self) -> str:
        """ String form of token that is used to indicate that a certain unit is to be perturbed. """
        raise NotImplementedError

    @property
    def mask_token_id(self) -> int:
        """ Integer form of token that is used to indicate that a certain unit is to be perturbed. """
        raise NotImplementedError

    @property
    def pad_token(self) -> str:
        raise NotImplementedError

    @property
    def pad_token_id(self) -> int:
        raise NotImplementedError

    @property
    def special_token_ids(self):
        raise NotImplementedError

    def from_internal(self, encoded_data: torch.Tensor,
                      skip_special_tokens: bool = True,
                      take_as_single_sequence: bool = False,
                      **kwargs) -> List[Union[str, Tuple[str, ...]]]:
        """ Convert from internal model representation to text. `kwargs` contains miscellaneous data that can be
        used as help for data reconstruction (e.g. attention mask)."""
        raise NotImplementedError

    def to_internal(self, text_data: List[Union[str, Tuple[str, ...]]],
                    pretokenized_text_data: Optional[Union[
                        List[List[str]],
                        List[Tuple[List[str], ...]]
                    ]] = None) -> Dict:
        """ Convert from text to internal model representation. Make sure to include 'perturbable_mask' in the
        returned dictionary."""
        raise NotImplementedError

    def tokenize(self, str_sequence: str) -> List[str]:
        """ Convert a sequence into its tokens."""
        raise NotImplementedError

    def encode_token(self, token: str) -> int:
        """ Turn a single token into its internal representation. """
        raise NotImplementedError

    def decode_token(self, token: int) -> str:
        """ Turn a single token from its internal representation to string"""
        raise NotImplementedError

    def convert_ids_to_tokens(self, ids: torch.Tensor) -> List[List[str]]:
        # TODO: remove?
        """ Convert integer-encoded tokens to str-encoded tokens, but keep them split."""
        raise NotImplementedError

    def score(self, input_ids: torch.Tensor, **aux_data):
        """ Obtain scores (e.g. probabilities for classes) for encoded data. Make sure to handle batching here.

        `aux_data` will contain all the auxiliary data required to do the modeling: one batch-first instance of the
        data, which will be repeated along first axis to match dimension of a batch of `input_ids`.
        """
        raise NotImplementedError


class DummySentiment(InterpretableModel):
    """ Dummy model that serves as an example (and testing): 2-word sentiment prediction (positive/negative). """
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
                      take_as_single_sequence: bool = False, **kwargs) -> List[str]:
        return [" ".join([self.id2tok[i] for i in sequence]) for sequence in encoded_data.tolist()]

    def to_internal(self, text_data: List[str], pretokenized_text_data=None) -> Dict:
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