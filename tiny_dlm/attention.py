from torch import nn
from torch.nn import functional as F

from tiny_dlm.rope import RotaryEmbedding


class SelfAttention(nn.Module):
    """
        Bidirectional Self-Attention Module
        ( why not causal like an LLM? a decision model reads the whole input at once and generates
          nothing, so every token may look at every other token, before and after it )

        Supports multi-head attention and grouped-query attention (Llama, Qwen2), where several query
        heads share the same key/value head (n_kv_head < n_head).

        `keep` marks what may be attended to: real tokens (not padding) inside a pair, or real
        candidates (not empty slots) when the candidates look at each other.

        `use_positions=False` turns RoPE off. Used for the candidate layer: the candidates are an
        unordered set, so their order must not change anything.

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
        self.n_head = config.n_head
        self.n_kv_head = config.kv_heads
        self.head_dim = config.head_size
        self.drop_rate = config.drop_rate
        q_dim, kv_dim = self.n_head * self.head_dim, self.n_kv_head * self.head_dim
        self.split_sizes = [q_dim, kv_dim, kv_dim]

        # query, key, value projections for all heads, in one batch
        self.c_attn = nn.Linear(config.n_embd, q_dim + 2 * kv_dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(q_dim, config.n_embd, bias=config.bias)
        self.res_drop = nn.Dropout(config.drop_rate)

        # the model shares one rotary table between all layers; standalone layers build their own
        self.rotary = None
        if use_positions and config.pos_encoding == "rope":
            self.rotary = rotary or RotaryEmbedding(self.head_dim, config.block_size, config.rope_theta)

    def forward(self, x, keep):
        B, T, _ = x.size()

        q, k, v = self.c_attn(x).split(self.split_sizes, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv, T, hs)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv, T, hs)

        if self.rotary is not None:  # positions go into q and k only, never into v
            q, k = self.rotary(q, k)

        if self.n_kv_head != self.n_head:  # grouped-query attention: share each kv head between query heads
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        # every query may look at every kept key: (B, 1, 1, T) broadcasts over heads and queries
        mask = keep[:, None, None, :]

        # flash attention when available: softmax(q k^T / sqrt(hs)) v
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.drop_rate if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)  # re-assemble all heads

        return self.res_drop(self.c_proj(y))
