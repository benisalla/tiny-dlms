import torch
from torch import nn


def rotate_half(X):
    """[x1, x2] -> [-x2, x1] on the last dimension."""
    x1, x2 = X.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
        Rotary Position Embedding (RoPE, https://arxiv.org/abs/2104.09864).
        Instead of adding a position vector to the input, Q and K are rotated by an angle that grows
        with the position, so Q·K only depends on the *distance* between two tokens.
        Used by: Llama, Qwen2, ModernBERT.

            m      token position             0, 1, 2, ...
            i      pair index inside a head   0, 1, ..., h/2 - 1
            theta  rotation angle             theta[m, i] = m / base^(2i / h)

            rotate(X) = X * cos(theta) + rotate_half(X) * sin(theta)

        Tensor          Type            Shape
        ===========================================================================
        Q, K            float           (b, nh, s, h)
        ===========================================================================
    """

    def __init__(self, h, max_positions, base=10000.0):
        super().__init__()
        assert h % 2 == 0, "RoPE needs an even h (dimensions per head)"
        self.h, self.base, self.max_positions = h, base, max_positions
        self.reset_buffers()

    def reset_buffers(self, device=None):
        """(Re)compute the cos/sin tables, they are not saved in checkpoints."""
        i = torch.arange(self.h // 2, device=device).float()  # pair index
        freq = 1.0 / self.base ** (2 * i / self.h)  # (h/2,)
        m = torch.arange(self.max_positions, device=device).float()  # token positions
        theta = torch.outer(m, freq)  # (max_positions, h/2)
        theta = torch.cat((theta, theta), dim=-1)  # (max_positions, h): both halves of a pair share an angle
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, Q, K):
        s = Q.size(-2)
        cos = self.cos[:s].to(Q.dtype)  # (s, h)
        sin = self.sin[:s].to(Q.dtype)
        return Q * cos + rotate_half(Q) * sin, K * cos + rotate_half(K) * sin
