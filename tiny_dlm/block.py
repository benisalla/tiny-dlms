from torch import nn

from tiny_dlm.attention import SelfAttention
from tiny_dlm.mlp import MLP
from tiny_dlm.norm import build_norm


class EBlock(nn.Module):
    """
        Transformer (Encoder) Block: communication layer (att mechanism) + computation layer (feedforward).
        Pre-normalisation, as in GPT-2 and every model after it.

            x = x + attn(ln_1(x), keep)
            x = x + mlp(ln_2(x))

        Same block as tiny-llm's DBlock, except attention is bidirectional (Encoder) instead of
        causal (Decoder), and it takes a `keep` mask for padding.

        Tensor          Type            Shape
        ===========================================================================
        input           float           (batch_size, seq_len, n_embd)
        keep            bool            (batch_size, seq_len)
        ---------------------------------------------------------------------------
        output          float           (batch_size, seq_len, n_embd)
        ===========================================================================
    """

    def __init__(self, config, rotary=None, use_positions=True):
        super().__init__()
        self.ln_1 = build_norm(config)
        self.attn = SelfAttention(config, rotary, use_positions)
        self.ln_2 = build_norm(config)
        self.mlp = MLP(config)

    def forward(self, x, keep):
        x = x + self.attn(self.ln_1(x), keep)
        return x + self.mlp(self.ln_2(x))
