from typing import List, Union, Tuple, Optional, Dict

import torch


# TODO: TokenizerBase in some utils package? tokenization methods are likely gonna be same/similar in model and generator
class InterpretableModel:
    def __init__(self, max_seq_len: int, batch_size: int = 8, device: str = "cuda"):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        assert device in ["cpu", "cuda"]
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Device is set to 'cuda', but no CUDA device could be found. If you want to run the model "
                             "on CPU, set device to 'cpu'")
        self.device = torch.device(device)

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

    def from_internal(self, encoded_data, skip_special_tokens: bool = True, take_as_single_sequence: bool = False,
                      return_tokens=False, **kwargs) -> List[Union[str, Tuple[str, ...],
                                                                   List[str], Tuple[List[str], ...]]]:
        """ Convert from internal model representation to text. `kwargs` contains miscellaneous data that can be
        used as help for data reconstruction (e.g. attention mask)."""
        raise NotImplementedError

    def to_internal(self, text_data: Union[List[str], List[Tuple[str, ...]],
                                           List[List[str]], List[Tuple[List[str], ...]]],
                    is_split_into_units: Optional[bool] = False,
                    allow_truncation: Optional[bool] = True) -> Dict:
        """ Convert from text to internal generator representation.
        `allow_truncation` specifies whether overflowing tokens (past max_seq_len) are allowed to be dropped. """
        raise NotImplementedError

    def score(self, input_ids: torch.Tensor, **aux_data):
        """ Obtain scores (e.g. probabilities for classes) for encoded data. Make sure to handle batching here.

        `aux_data` will contain all the auxiliary data required to do the modeling: one batch-first instance of the
        data, which will be repeated along first axis to match dimension of a batch of `input_ids`.
        """
        raise NotImplementedError
