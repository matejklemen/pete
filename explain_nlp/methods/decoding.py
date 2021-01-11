import torch
import torch.nn.functional as F


def greedy_decoding(logits: torch.Tensor):
    return torch.argmax(logits, dim=-1, keepdim=True)


def top_p_decoding(logits: torch.Tensor, top_p: float):
    # TODO: instead of keeping all logits, performing softmax over them etc., we could only keep the valid ones
    logits = top_p_filtering(logits, top_p=top_p)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


def top_k_decoding(logits: torch.Tensor, top_k: int):
    top_logits, top_i = torch.topk(logits, k=top_k)
    selected = torch.multinomial(F.softmax(top_logits, dim=-1), num_samples=1)
    return top_i[torch.arange(logits.shape[0]).unsqueeze(1), selected]


def top_p_filtering(logits, top_p):
    """ Sets tokens that go beyond top_p cumulative probability to unsamplable (logit = -inf). """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

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


def top_k_filtering(logits, top_k):
    """ Makes all but the top K tokens unsamplable (logit = -inf). """
    mask = torch.ones_like(logits, dtype=torch.bool)
    top_logits, top_i = torch.topk(logits, k=top_k)
    mask[torch.arange(logits.shape[0]).unsqueeze(1), top_i] = False

    logits_copy = logits.clone()
    logits_copy[mask] = -float("inf")

    return logits_copy


if __name__ == "__main__":
    probas = torch.tensor([
        [0.05, 0.1, 0.05, 0.03, 0.5, 0.27],
        [0.05, 0.27, 0.1, 0.05, 0.03, 0.5]
    ])  # B, |V|
    logprobas = torch.log(probas)  # B, |V|

    filtered_logprobas = top_k_filtering(logprobas, top_k=2)
    print("Original:")
    print(logprobas)
    print("Filtered:")
    print(filtered_logprobas)
