from typing import List, Union, Tuple, Dict, Optional

import torch

from explain_nlp.methods.decoding import top_p_filtering, top_k_filtering


class SampleGenerator:
    def __init__(self, max_seq_len: int, batch_size: int = 8, device: str = "cuda",
                 strategy: str = "top_p", top_p: Optional[float] = None, top_k: Optional[int] = 5,
                 threshold: Optional[float] = 0.1):
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.device = device
        self.strategy = strategy
        self.top_p = top_p
        self.top_k = top_k
        self.threshold = threshold

        assert self.device in ["cpu", "cuda"]
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Device is set to 'cuda', but no CUDA device could be found. If you want to run the model "
                             "on CPU, set device to 'cpu'")

        if self.strategy == "top_p":
            assert self.top_p is not None
            self.filtering_strategy = lambda logits: top_p_filtering(logits, top_p=self.top_p)
        elif self.strategy == "top_k":
            assert self.top_k is not None
            self.filtering_strategy = lambda logits: top_k_filtering(logits, top_k=self.top_k)
        elif self.strategy == "threshold":
            raise NotImplementedError(f"Unimplemented (but planned) strategy: '{self.strategy}'")  # TODO
        else:
            raise NotImplementedError(f"Unsupported filtering strategy: '{strategy}'")

    @property
    def mask_token(self) -> str:
        raise NotImplementedError

    @property
    def mask_token_id(self) -> int:
        raise NotImplementedError

    def convert_ids_to_tokens(self, ids: torch.Tensor) -> List[List[str]]:
        """ Convert integer-encoded tokens to str-encoded tokens, but keep them split."""
        raise NotImplementedError

    def from_internal(self, encoded_data: torch.Tensor, skip_special_tokens=True) -> List[Union[str, Tuple[str, ...]]]:
        """ Convert from internal generator representation to text."""
        raise NotImplementedError

    def to_internal(self, text_data: List[Union[str, Tuple[str, ...]]]) -> Dict:
        """ Convert from text to internal generator representation.
        Make sure to include 'input_ids', 'perturbable_mask' and 'aux_data' in the returned dictionary."""
        raise NotImplementedError

    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor, num_samples: int,
                 label: int, **aux_data) -> Dict:
        raise NotImplementedError

    def generate_masked_samples(self, masked_input_ids: torch.Tensor,
                                **generation_kwargs):
        raise NotImplementedError
