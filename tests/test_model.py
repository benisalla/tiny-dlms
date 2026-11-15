import pytest
import torch

from conftest import tiny_config
from tiny_dlm import Decision, TinyDLM, collate


def test_shapes_and_empty_slots_get_zero_probability(tiny_model, tokenizer, decisions):
    idx, cand_mask, targets = collate(decisions, tokenizer)
    logits, loss = tiny_model(idx, cand_mask, targets)
    probs = torch.softmax(logits, dim=-1)

    assert logits.shape == (3, 4)
    assert probs[0, 2:].sum() == 0  # the first decision has only 2 candidates
    assert probs[2, 3] == 0  # the third has 3
    assert torch.allclose(probs.sum(-1), torch.ones(3))
    assert torch.isfinite(loss)


def test_no_targets_no_loss(tiny_model, tokenizer, decisions):
    idx, cand_mask, _ = collate(decisions, tokenizer)
    _, loss = tiny_model(idx, cand_mask)
    assert loss is None


def test_gradients_are_finite_with_empty_slots(tokenizer, decisions):
    model = TinyDLM(tiny_config(vocab_size=len(tokenizer)))
    _, loss = model(*collate(decisions, tokenizer))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_shuffling_candidates_only_shuffles_scores(tiny_model, tokenizer):
    cands = ["go to sleep", "wear a coat", "eat some food", "read a book"]
    perm = [2, 0, 3, 1]
    a = collate([Decision("i am hungry", cands, cands[:1])], tokenizer)
    b = collate([Decision("i am hungry", [cands[i] for i in perm], cands[:1])], tokenizer)
    _, pa = tiny_model.decide(*a[:2])
    _, pb = tiny_model.decide(*b[:2])
    assert torch.allclose(pb[0], pa[0, perm], atol=1e-5)


def test_word_order_matters(tiny_model, tokenizer):
    # RoPE gives the pair encoder a sense of order
    a = collate([Decision("it is raining", ["take an umbrella", "eat some food"], ["take an umbrella"])], tokenizer)
    b = collate([Decision("raining is it", ["take an umbrella", "eat some food"], ["take an umbrella"])], tokenizer)
    assert not torch.allclose(tiny_model(*a[:2])[0], tiny_model(*b[:2])[0])


@pytest.mark.parametrize("overrides", [
    dict(),                                                                    # Llama-style (default)
    dict(norm_type="layernorm", pos_encoding="learned", gated_mlp=False, activation="gelu_tanh", bias=True),  # GPT-2
    dict(n_kv_head=2),                                                         # grouped-query attention
])
def test_every_design_runs(tokenizer, decisions, overrides):
    model = TinyDLM(tiny_config(vocab_size=len(tokenizer), **overrides))
    logits, loss = model(*collate(decisions, tokenizer))
    assert logits.shape == (3, 4) and torch.isfinite(loss)


def test_decide_with_temperature_samples_only_real_candidates(tiny_model, tokenizer, decisions):
    idx, cand_mask, _ = collate(decisions, tokenizer)
    for _ in range(20):
        choice, _ = tiny_model.decide(idx, cand_mask, temperature=1.0)
        assert cand_mask[torch.arange(3), choice].all()


def test_optimizer_decays_only_matrices(tiny_model):
    opt = tiny_model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    decay, no_decay = opt.param_groups
    assert all(p.dim() >= 2 for p in decay["params"]) and decay["weight_decay"] == 0.1
    assert all(p.dim() < 2 for p in no_decay["params"]) and no_decay["weight_decay"] == 0.0


def test_too_long_input_is_refused(tokenizer):
    model = TinyDLM(tiny_config(vocab_size=len(tokenizer), block_size=8))
    idx, cand_mask, _ = collate([Decision("i am hungry and i am tired", ["go to sleep"], ["go to sleep"])], tokenizer)
    with pytest.raises(AssertionError, match="Max sequence length"):
        model(idx, cand_mask)


# ---------------------------------------------------------------- pre-training mode


def test_masked_word_prediction_mode(tiny_model, tokenizer):
    idx = torch.tensor([[tokenizer.cls_id] + tokenizer.encode("it is") + [tokenizer.mask_id] + tokenizer.encode(".")])
    targets = torch.full_like(idx, -1)
    targets[0, 3] = tokenizer.stoi["raining"]
    logits, loss = tiny_model(idx, targets=targets)
    assert logits.shape == (1, 5, len(tokenizer))
    assert abs(loss.item() - torch.log(torch.tensor(float(len(tokenizer)))).item()) < 1.0  # ~random at init


def test_mlm_head_is_tied_to_token_embeddings(tiny_model):
    assert tiny_model.mlm_head.weight is tiny_model.transformer.wte.weight
