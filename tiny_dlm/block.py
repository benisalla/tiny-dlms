from torch import nn

from tiny_dlm.attention import SelfAttention
from tiny_dlm.mlp import MLP
from tiny_dlm.norm import build_norm


class EBlock(nn.Module):
    """
        Transformer (Encoder) Block: communication layer (att mechanism) + computation layer (feedforward).
        Pre-normalisation, as in GPT-2 and every model after it.

            X = X + attn(ln_1(X), keep)
            X = X + mlp(ln_2(X))

        Same block as tiny-llm's DBlock, except attention is bidirectional (Encoder) instead of
        causal (Decoder), and it takes a `keep` mask for padding.

        Tensor          Type            Shape
        ===========================================================================
        input           float           (b, s, e)      b sequences, s tokens, e embedding size
        keep            bool            (b, s)
        ---------------------------------------------------------------------------
        output          float           (b, s, e)      b sequences, s tokens, e embedding size
        ===========================================================================
    """

    def __init__(self, config, rotary=None, use_positions=True):
        super().__init__()
        self.ln_1 = build_norm(config)
        self.attn = SelfAttention(config, rotary, use_positions)
        self.ln_2 = build_norm(config)
        self.mlp = MLP(config)

    def forward(self, X, keep):
        X = X + self.attn(self.ln_1(X), keep)
        return X + self.mlp(self.ln_2(X))
