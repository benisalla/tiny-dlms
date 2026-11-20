from torch import nn
from torch.nn import functional as F

from tiny_dlm.rope import RotaryEmbedding


class SelfAttention(nn.Module):
    """
        Bidirectional Self-Attention Module
        ( why not causal like an LLM? a decision model reads the whole input at once and generates
          nothing, so every token may look at every other token, before and after it )

        Supports multi-head attention and grouped-query attention (Llama, Qwen2), where several query
        heads share the same key/value head (nkv < nh).

        `keep` marks what may be attended to: real tokens (not padding) inside a pair, or real
        candidates (not empty slots) when the candidates look at each other.

        `use_positions=False` turns RoPE off. Used for the candidate layer: the candidates are an
        unordered set, so their order must not change anything.

            b batch, s sequence length, e embedding size, nh heads, nkv key/value heads, h size per head

        Tensor          Type            Shape
        ===========================================================================
        X               float           (b, s, e)
        keep            bool            (b, s)
        Q               float           (b, nh, s, h)
        K, V            float           (b, nkv, s, h) -> repeated to (b, nh, s, h)
        ---------------------------------------------------------------------------
        output          float           (b, s, e)
        ===========================================================================
    """

    def __init__(self, config, rotary=None, use_positions=True):
        super().__init__()
        self.nh = config.n_head
        self.nkv = config.kv_heads
        self.h = config.head_size
        self.drop_rate = config.drop_rate
        self.split_sizes = [self.nh * self.h, self.nkv * self.h, self.nkv * self.h]

        # Q, K, V projections for all heads, in one matrix
        self.c_attn = nn.Linear(config.n_embd, sum(self.split_sizes), bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(self.nh * self.h, config.n_embd, bias=config.bias)
        self.res_drop = nn.Dropout(config.drop_rate)

        # the model shares one rotary table between all layers; standalone layers build their own
        self.rotary = None
        if use_positions and config.pos_encoding == "rope":
            self.rotary = rotary or RotaryEmbedding(self.h, config.block_size, config.rope_base)

    def forward(self, X, keep):
        b, s, _ = X.size()

        Q, K, V = self.c_attn(X).split(self.split_sizes, dim=2)
        Q = Q.view(b, s, self.nh, self.h).transpose(1, 2)  # (b, nh, s, h)
        K = K.view(b, s, self.nkv, self.h).transpose(1, 2)  # (b, nkv, s, h)
        V = V.view(b, s, self.nkv, self.h).transpose(1, 2)  # (b, nkv, s, h)

        if self.rotary is not None:  # positions go into Q and K only, never into V
            Q, K = self.rotary(Q, K)

        if self.nkv != self.nh:  # grouped-query attention: each K/V head is shared by nh / nkv query heads
            K = K.repeat_interleave(self.nh // self.nkv, dim=1)  # (b, nh, s, h)
            V = V.repeat_interleave(self.nh // self.nkv, dim=1)

        # every query may look at every kept key: (b, 1, 1, s) broadcasts over heads and queries
        mask = keep[:, None, None, :]

        # flash attention when available: softmax(Q K^T / sqrt(h)) V
        context = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask,
                                                 dropout_p=self.drop_rate if self.training else 0.0)  # (b, nh, s, h)
        context = context.transpose(1, 2).contiguous().view(b, s, self.nh * self.h)  # (b, s, e)

        return self.res_drop(self.c_proj(context))
