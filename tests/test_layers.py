import pytest
import torch

from conftest import tiny_config
from tiny_dlm.attention import SelfAttention
from tiny_dlm.block import EBlock
from tiny_dlm.mlp import MLP
from tiny_dlm.norm import LayerNorm, RMSNorm, build_norm
from tiny_dlm.rope import RotaryEmbedding


def test_rmsnorm_gives_unit_rms():
    x = torch.randn(4, 16) * 7
    y = RMSNorm(16)(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4), atol=1e-3)


def test_build_norm_follows_config():
    assert isinstance(build_norm(tiny_config(norm_type="rmsnorm")), RMSNorm)
    assert isinstance(build_norm(tiny_config(norm_type="layernorm")), LayerNorm)


def test_rope_depends_only_on_relative_position():
    rope = RotaryEmbedding(h=8, max_positions=32)
    q_vec, k_vec = torch.randn(8), torch.randn(8)

    def score(m, n):
        # q at position m, k at position n -> rotated dot product
        q = torch.zeros(1, 1, 32, 8)
        k = torch.zeros(1, 1, 32, 8)
        q[0, 0, m], k[0, 0, n] = q_vec, k_vec
        q, k = rope(q, k)
        return (q[0, 0, m] * k[0, 0, n]).sum()

    assert torch.allclose(score(2, 5), score(10, 13), atol=1e-5)  # same distance (3)
    assert not torch.allclose(score(2, 5), score(2, 9), atol=1e-3)  # different distance


def test_rope_keeps_vector_length():
    rope = RotaryEmbedding(h=8, max_positions=16)
    q = torch.randn(2, 4, 16, 8)
    q_rot, _ = rope(q, q)
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5)  # a rotation only turns vectors


@pytest.mark.parametrize("n_kv_head", [0, 2])  # multi-head and grouped-query attention
def test_attention_ignores_padding(n_kv_head):
    torch.manual_seed(0)
    attn = SelfAttention(tiny_config(n_kv_head=n_kv_head)).eval()
    x = torch.randn(1, 6, 32)
    keep = torch.tensor([[True, True, True, True, False, False]])
    y1 = attn(x, keep)
    x2 = x.clone()
    x2[0, 4:] = torch.randn(2, 32)  # change only the padded positions
    y2 = attn(x2, keep)
    assert torch.allclose(y1[0, :4], y2[0, :4], atol=1e-6)  # real tokens did not notice


def test_attention_without_positions_is_order_free():
    torch.manual_seed(0)
    attn = SelfAttention(tiny_config(), use_positions=False).eval()
    x, keep = torch.randn(1, 5, 32), torch.ones(1, 5, dtype=torch.bool)
    perm = torch.tensor([3, 0, 4, 1, 2])
    assert torch.allclose(attn(x, keep)[0, perm], attn(x[:, perm], keep)[0], atol=1e-5)


@pytest.mark.parametrize("gated", [True, False])
def test_mlp_and_block_keep_shape(gated):
    cfg = tiny_config(gated_mlp=gated, activation="silu" if gated else "gelu_tanh")
    x = torch.randn(2, 7, 32)
    assert MLP(cfg)(x).shape == x.shape
    assert EBlock(cfg)(x, torch.ones(2, 7, dtype=torch.bool)).shape == x.shape


def test_rope_rotates_by_the_expected_angle():
    # h = 2 -> one pair (i = 0), frequency 1 / base^0 = 1, so theta = m: position m = 1 turns by 1 radian
    rope = RotaryEmbedding(h=2, max_positions=4)
    X = torch.zeros(1, 1, 4, 2)
    X[0, 0, 1] = torch.tensor([1.0, 0.0])
    Q, _ = rope(X, X)
    expected = torch.tensor([torch.cos(torch.tensor(1.0)), torch.sin(torch.tensor(1.0))])
    assert torch.allclose(Q[0, 0, 1], expected, atol=1e-6)
