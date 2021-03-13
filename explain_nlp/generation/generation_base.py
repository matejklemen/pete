from typing import List, Union, Tuple, Dict, Optional

import torch

from explain_nlp.methods.decoding import filter_factory


class SampleGenerator:
    def __init__(self, max_seq_len: int, batch_size: int = 8, device: str = "cuda",
                 strategy: Union[str, List] = "top_p", top_p: Optional[float] = None, top_k: Optional[int] = 5,
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
        self.filters = []
        strategy_list = strategy if isinstance(strategy, list) else [strategy]
        for curr_strategy in strategy_list:
            if isinstance(curr_strategy, str):
                self.filters.append(filter_factory(strategy=curr_strategy,
                                                   top_p=top_p, top_k=top_k, threshold=threshold))
            else:
                # custom function: (logits, **kwargs) => new_logits
                self.filters.append(curr_strategy)

    @property
    def mask_token(self) -> str:
        raise NotImplementedError

    @property
    def mask_token_id(self) -> int:
        raise NotImplementedError

    def from_internal(self, encoded_data, skip_special_tokens: bool = True, take_as_single_sequence: bool = False,
                      return_tokens: bool = False, **kwargs):
        """ Convert from internal generator representation to text."""
        raise NotImplementedError

    def to_internal(self, text_data: Union[List[str], List[Tuple[str, ...]],
                                           List[List[str]], List[Tuple[List[str], ...]]],
                    is_split_into_units: Optional[bool] = False) -> Dict:
        """ Convert from text to internal generator representation.
        Make sure to include 'input_ids', 'perturbable_mask' and 'aux_data' in the returned dictionary."""
        raise NotImplementedError

    def generate(self, input_ids: torch.Tensor, perturbable_mask: torch.Tensor, num_samples: int,
                 label: int, **aux_data) -> Dict:
        raise NotImplementedError

    def generate_masked_samples(self, input_ids: torch.Tensor,
                                generation_mask: torch.Tensor,
                                **generation_kwargs):
        raise NotImplementedError
