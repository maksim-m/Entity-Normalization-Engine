import torch
from torch import Tensor


def mean_pooling(token_embeddings: Tensor, attention_mask: Tensor):
    """
    Mean Pooling - Take attention mask into account for correct averaging
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
