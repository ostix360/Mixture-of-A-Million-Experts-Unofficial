import math
import warnings
from typing import Tuple

import einops
import numpy as np
import torch
from einops import einsum
from torch import nn
from torch.nn import functional as F


# copied from https://github.com/facebookresearch/XLM/blob/main/PKM-layer.ipynb
def get_uniform_keys(n_keys, dim, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    rng = np.random.RandomState(seed)
    bound = 1 / math.sqrt(dim)
    keys = rng.uniform(-bound, bound, (n_keys, dim))
    return keys.astype(np.float32)


class PEERBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_experts: int,
            num_experts_per_tok: int,
            num_heads: int,
            key_dim: int,
            batch_norm: bool = True,
            hidden_act: nn.Module = nn.ReLU(),
            glu: bool = False,
    ):
        """
        Initialize the PEER block
        Args:
            d_model: The dimension of the model
            num_experts: The number of experts
                            it's will be squared during initialisation to get the correct number of experts and keys
            num_experts_per_tok: The number of experts per token (top_k)
            num_heads: The number of heads
            key_dim: The dimension of each key
            batch_norm: Whether to use batch normalization during training (default: True)
            hidden_act: The activation function to use (default: ReLU)
        """
        super().__init__()
        self.hidden_size = d_model
        self.num_experts = num_experts ** 2
        self.num_experts_per_tok = num_experts_per_tok
        self.heads = num_heads
        assert math.sqrt(self.num_experts).is_integer(), '`num_experts` needs to be a perfect square'
        self.num_keys = int(math.sqrt(self.num_experts))
        self.k_dim = key_dim
        self.glu = glu
        self.w_up_embed = nn.Embedding(self.num_experts, self.hidden_size)
        self.w_down_embed = nn.Embedding(self.num_experts, self.hidden_size)
        self.w_gate_embed = nn.Embedding(self.num_experts, self.hidden_size)
        self.act_fn = hidden_act
        self.initialize_keys()
        # query network # copied from https://github.com/facebookresearch/XLM/blob/main/PKM-layer.ipynb
        self.query_proj = nn.Sequential(*filter(None, [
            nn.Linear(self.hidden_size, self.heads * self.k_dim, bias=True),
            nn.BatchNorm1d(self.heads * self.k_dim) if batch_norm else None
        ]))
        if batch_norm:  # copied from https://github.com/facebookresearch/XLM/blob/main/PKM-layer.ipynb
            warnings.warn("WARNING: Applying batch normalization to queries improves the performance "
                          "and memory usage. But if you use it, be sure that you use batches of "
                          "sentences with the same size at training time (i.e. without padding). "
                          "Otherwise, the padding token will result in incorrect mean/variance "
                          "estimations in the BatchNorm layer.\n")

    # copied from https://github.com/facebookresearch/XLM/blob/main/PKM-layer.ipynb
    def initialize_keys(self):
        """
        Create two subkey sets per head.
        `self.keys` is of shape (heads, 2, n_keys, k_dim // 2)
        """
        half = self.k_dim // 2
        keys = nn.Parameter(torch.from_numpy(np.array([
            get_uniform_keys(self.num_keys, half, seed=(2 * i + j))
            for i in range(self.heads)
            for j in range(2)
        ])).view(self.heads, 2, self.num_keys, half))
        self.keys = nn.Parameter(keys)

    def _get_indices(self, query, subkeys, top_k):
        """
        Generate scores and indices for a specific head.
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim
        bs = query.size(0)
        knn = top_k
        half = self.k_dim // 2
        n_keys = len(subkeys[0])

        # split query for product quantization
        q1 = query[:, :half]  # (bs,half)
        q2 = query[:, half:]  # (bs,half)

        # compute indices with associated scores
        scores1 = F.linear(q1, subkeys[0], bias=None)  # (bs,n_keys)
        scores2 = F.linear(q2, subkeys[1], bias=None)  # (bs,n_keys)
        scores1, indices1 = scores1.topk(knn, dim=1)  # (bs,knn)
        scores2, indices2 = scores2.topk(knn, dim=1)  # (bs,knn)

        # cartesian product on best candidate keys
        all_scores = (
                scores1.view(bs, knn, 1).expand(bs, knn, knn) +
                scores2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)  # (bs,knn**2)
        all_indices = (
                indices1.view(bs, knn, 1).expand(bs, knn, knn) * n_keys +
                indices2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)  # (bs,knn**2)

        # select best scores with associated indices
        scores, best_indices = torch.topk(all_scores, k=knn, dim=1)  # (bs,knn)
        indices = all_indices.gather(1, best_indices)  # (bs,knn)

        assert scores.shape == indices.shape == (bs, knn)
        return indices, scores, all_scores

    def get_indices(self, query, bsz, seq_len):
        """
        Generate scores and indices.
        Reshape to the correct shape indices and scores
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim
        query = query.view(-1, self.heads, self.k_dim)
        bs = len(query)
        outputs = [self._get_indices(query[:, i], self.keys[i], self.num_experts_per_tok) for i in range(self.heads)]
        s = torch.cat([s.view(bs, 1, self.num_experts_per_tok) for _, s, _ in outputs], 1)  # (bs,heads,knn)
        i = torch.cat([i.view(bs, 1, self.num_experts_per_tok) for i, _, _ in outputs], 1)  # (bs,heads,knn)
        all_scores = torch.cat([a.view(bs, 1, self.num_experts_per_tok ** 2) for _, _, a in outputs], 1)
        i = einops.rearrange(i, "(b t) h k -> b t h k", b=bsz, t=seq_len)
        s = einops.rearrange(s, "(b t) h k -> b t h k", b=bsz, t=seq_len)
        return i, s, all_scores.view(-1, self.num_experts_per_tok ** 2)

    # Algorithm from the paper
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Args:
            hidden_states: the input of the block

        Returns:
            the output of the block and the scores to calculate the loss
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        indices, scores, all_scores = self.get_indices(
            self.query_proj(hidden_states).view(batch_size * sequence_length * self.heads, self.k_dim),
            batch_size,
            sequence_length,
        )
        w_up = self.w_up_embed(indices)
        if self.glu:
            w_down = self.w_down_embed(indices)
        w_gate = self.w_gate_embed(indices)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        gate = einsum(hidden_states, w_gate, "b t d , b t h k d -> b t h k ")
        hidden_states = einsum(hidden_states, w_down, "b t d , b t h k d -> b t h k ")
        hidden_states = self.act_fn(hidden_states) * gate if self.glu else self.act_fn(hidden_states)
        hidden_states = hidden_states * F.softmax(scores, dim=-1)
        hidden_states = einsum(hidden_states, w_up, "b t h k, b t h k d -> b t d")
        return hidden_states, all_scores
