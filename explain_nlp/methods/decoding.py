from typing import Optional

import torch
import torch.nn.functional as F


def greedy_decoding(logits: torch.Tensor, ensure_diff_from: Optional[torch.Tensor] = None):
    if ensure_diff_from is not None:
        batch_index = torch.arange(logits.shape[0])
        if ensure_diff_from.dim() == 2:
            batch_index = batch_index.unsqueeze(1)

        logits[batch_index, ensure_diff_from] = -float("inf")

    return torch.argmax(logits, dim=-1, keepdim=True)


def top_p_decoding(logits: torch.Tensor, top_p: float, ensure_diff_from: Optional[torch.Tensor] = None):
    if ensure_diff_from is not None:
        batch_index = torch.arange(logits.shape[0])
        if ensure_diff_from.dim() == 2:
            batch_index = batch_index.unsqueeze(1)

        logits[batch_index, ensure_diff_from] = -float("inf")

    logits = top_p_filtering(logits, top_p=top_p)

    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


def top_p_filtering(logits, top_p):
    """ Sets tokens that go beyond top_p cumulative probability to 'unsampleable' (logit = -inf). """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    _indices_to_remove = torch.nonzero(sorted_indices_to_remove, as_tuple=False)
    _row = _indices_to_remove[:, 0]
    _token = sorted_indices[_row, _indices_to_remove[:, 1]]
    logits[_row, _token] = -float("inf")

    return logits