import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """
        Layer Normalization with an optional bias (PyTorch's nn.LayerNorm doesn't support bias=False).
        Used by: GPT-2, BERT.

        y = (X - mean(X)) / sqrt(var(X) + eps) * weight + bias

        Tensor          Type            Shape
        ===========================================================================
        input           float           (b, s, e)      b sequences, s tokens, e embedding size
        ---------------------------------------------------------------------------
        output          float           (b, s, e)      b sequences, s tokens, e embedding size
        ===========================================================================
    """

    def __init__(self, ndim, bias, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    """
        Root Mean Square normalization: no mean subtraction and no bias, a bit cheaper than LayerNorm
        and works as well.   Used by: Llama, Qwen2, ModernBERT.

        y = X / sqrt(mean(X^2) + eps) * weight

        Tensor          Type            Shape
        ===========================================================================
        input           float           (..., n_embd)
        ---------------------------------------------------------------------------
        output          float           (..., n_embd)
        ===========================================================================
    """

    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, X):
        dtype = X.dtype
        X = X.float()  # normalise in float32 for stability under mixed precision
        X = X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * X.to(dtype)


def build_norm(config):
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.n_embd, eps=config.norm_eps)
    return LayerNorm(config.n_embd, bias=config.bias, eps=config.norm_eps)
