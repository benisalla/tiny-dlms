import torch
from torch import nn


def rotate_half(x):
    """[x1, x2] -> [-x2, x1] on the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
        Rotary Position Embedding (RoPE, https://arxiv.org/abs/2104.09864).
        Instead of adding a position vector to the input, q and k are rotated by an angle that grows
        with the position, so q·k only depends on the *distance* between two tokens.
        Used by: Llama, Qwen2, ModernBERT.

            angle(pos, j) = pos * theta^(-2j / head_dim)
            rotate(x)     = x * cos(angle) + rotate_half(x) * sin(angle)

        Tensor          Type            Shape
        ===========================================================================
        q, k            float           (batch_size, n_heads, seq_len, head_dim)
        ===========================================================================
    """

    def __init__(self, head_dim, max_positions, theta=10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head_dim"
        self.head_dim, self.theta, self.max_positions = head_dim, theta, max_positions
        self.reset_buffers()

    def reset_buffers(self, device=None):
        """(Re)compute the cos/sin tables, they are not saved in checkpoints."""
        arange = torch.arange(0, self.head_dim, 2, dtype=torch.int64, device=device).float()
        inv_freq = 1.0 / (self.theta ** (arange / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(self.max_positions, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # (T, head_dim / 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, q, k):
        T = q.size(-2)
        cos = self.cos[:T].to(q.dtype)
        sin = self.sin[:T].to(q.dtype)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
