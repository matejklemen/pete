from typing import List

import torch


def greedy_decoding(logits: torch.Tensor):
    return torch.argmax(logits, dim=-1, keepdim=True)


def top_p_decoding(logits: torch.Tensor, top_p: float):
    logits = top_p_filtering(logits, top_p=top_p)
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


def top_k_decoding(logits: torch.Tensor, top_k: int):
    top_logits, top_i = torch.topk(logits, k=top_k)
    selected = torch.multinomial(torch.softmax(top_logits, dim=-1), num_samples=1)
    return top_i[torch.arange(logits.shape[0]), torch.flatten(selected)].unsqueeze(1)


def filter_factory(strategy="top_p", top_p=0.9, top_k=5,
                   allowed_values: List[torch.Tensor] = None):
    if strategy is None:
        # no-op, used to simplify code
        def strategy_fn(logits, **strategy_kwargs):
            return logits
    elif strategy == "top_p":
        def strategy_fn(logits, **strategy_kwargs):
            return top_p_filtering(logits, top_p=top_p)
    elif strategy == "top_k":
        def strategy_fn(logits, **strategy_kwargs):
            return top_k_filtering(logits, top_k=top_k)
    elif strategy == "unique":
        def strategy_fn(logits, orig_values, **strategy_kwargs):
            return filter_unique(logits, orig_values=orig_values)
    elif strategy == "allowed":
        assert allowed_values is not None

        def strategy_fn(logits, curr_position, **strategy_kwargs):
            return filter_allowed(logits, allowed_values=allowed_values[curr_position])
    else:
        raise NotImplementedError(f"Unrecognized strategy: '{strategy}'")

    return strategy_fn


def top_p_filtering(logits, top_p, **kwargs):
    """ Sets tokens that go beyond top_p cumulative probability to unsamplable (logit = -inf). """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    _indices_to_remove = torch.nonzero(sorted_indices_to_remove, as_tuple=False)
    _row = _indices_to_remove[:, 0]
    _token = sorted_indices[_row, _indices_to_remove[:, 1]]
    logits_copy = logits.clone()
    logits_copy[_row, _token] = -float("inf")

    return logits_copy


def top_k_filtering(logits, top_k, **kwargs):
    """ Makes all but the top K tokens unsamplable (logit = -inf). """
    mask = torch.ones_like(logits, dtype=torch.bool)
    top_logits, top_i = torch.topk(logits, k=top_k)
    mask[torch.arange(logits.shape[0]).unsqueeze(1), top_i] = False

    logits_copy = logits.clone()
    logits_copy[mask] = -float("inf")

    return logits_copy


def filter_unique(logits, orig_values, **kwargs):
    """ Makes original values unsamplable.
    Args:
        logits:
            Predicted logits for token at `position`. Shape: [batch_size, vocab_size]
        orig_values:
            Values present in the original input. Shape: [batch_size]
    """
    logits[torch.arange(orig_values.shape[0]), orig_values] = -float("inf")
    return logits


def filter_allowed(logits, allowed_values, **kwargs):
    """ Makes all but allowed values unsamplable.

    Args:
        logits:
            Predicted logits for token at `position`. Shape: [batch_size, vocab_size]
        allowed_values:
            Values that should remain possible to sample. Shape: [num_allowed_values]
    """
    unsamplable_mask = torch.ones(logits.shape[1], dtype=torch.bool)
    unsamplable_mask[allowed_values] = False

    logits[:, unsamplable_mask] = -float("inf")
    return logits


if __name__ == "__main__":
    probas = torch.tensor([
        [0.05, 0.1, 0.05, 0.03, 0.5, 0.27],
        [0.05, 0.27, 0.1, 0.05, 0.03, 0.5]
    ])  # B, |V|
    logprobas = torch.log(probas)  # B, |V|

    filtered_logprobas = top_k_decoding(logprobas, top_k=2)
    print("Original:")
    print(logprobas)
    print("Filtered:")
    print(filtered_logprobas)
