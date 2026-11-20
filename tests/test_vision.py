import numpy as np
import torch

from conftest import tiny_config
from tiny_dlm import WordTokenizer
from tiny_dlm.car import OPTION_ACTIONS, OPTION_TEXTS, STEERS, nearest_option, preprocess
from tiny_dlm.vision import ImageDLM, PatchEmbedding, encode_options, load_image_dlm


def image_model(seed=0, **overrides):
    tok = WordTokenizer.from_texts(OPTION_TEXTS)
    torch.manual_seed(seed)
    cfg = tiny_config(vocab_size=len(tok), block_size=48, pad_id=tok.pad_id, sep_id=tok.sep_id, **overrides)
    return ImageDLM(cfg).eval(), tok


def batch(tok, B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    images = torch.randint(0, 256, (B, 3, 48, 48), dtype=torch.uint8, generator=g)
    idx = encode_options(OPTION_TEXTS, tok).expand(B, -1, -1)
    return images, idx, torch.ones(B, len(OPTION_TEXTS), dtype=torch.bool)


def test_patches_cover_the_image_in_order():
    pe = PatchEmbedding(image_size=48, patch_size=8, channels=3, n_embd=16)
    assert pe.n_patches == 36
    torch.nn.init.zeros_(pe.pos)
    img = torch.zeros(1, 3, 48, 48, dtype=torch.uint8)
    img[0, :, 8:16, 0:8] = 255  # the first patch of the second row = patch 6
    out = pe(img)
    changed = (out[0] - out[0, 0]).abs().sum(-1) > 1e-6
    assert changed.nonzero().flatten().tolist() == [6]


def test_shapes_probabilities_and_initial_loss():
    model, tok = image_model()
    images, idx, mask = batch(tok)
    targets = torch.eye(15)[torch.tensor([2, 7, 13])]  # a different right option per image
    logits, loss = model(images, idx, mask, targets)
    assert logits.shape == (3, 15)
    assert abs(loss.item() - np.log(15)) < 0.2  # random guessing at init
    _, probs = model.decide(images, idx, mask)
    assert torch.allclose(probs.sum(-1), torch.ones(3))


def test_shuffling_options_only_shuffles_probabilities_and_any_k_works():
    model, tok = image_model()
    images, idx, mask = batch(tok, B=1)
    _, p = model.decide(images, idx, mask)
    perm = torch.randperm(15)
    _, q = model.decide(images, idx[:, perm], mask)
    assert torch.allclose(p[0, perm], q[0], atol=1e-5)
    _, p4 = model.decide(images, idx[:, :4], mask[:, :4])  # only 4 options: still a distribution
    assert p4.shape == (1, 4) and torch.allclose(p4.sum(), torch.tensor(1.0))


def test_checkpoint_round_trip(tmp_path):
    from tiny_dlm.checkpoint import save_checkpoint
    model, tok = image_model()
    path = tmp_path / "car.pt"
    save_checkpoint(path, model, extra={"tokenizer": tok.to_dict(), "option_texts": OPTION_TEXTS,
                                        "image_args": model.image_args})
    loaded, tok2, texts, _ = load_image_dlm(path)
    images, idx, mask = batch(tok)
    assert texts == OPTION_TEXTS and tok2.itos == tok.itos
    assert torch.allclose(model(images, idx, mask)[0], loaded(images, idx, mask)[0])


def test_car_options_and_rounding():
    assert len(OPTION_TEXTS) == 2 * len(STEERS) + 1 == 15
    for k, action in enumerate(OPTION_ACTIONS):
        assert nearest_option(action) == k  # every option rounds to itself
    assert OPTION_TEXTS[nearest_option(np.array([-0.3, 0.5, 0.0]))] == "steer slight left and accelerate"
    assert OPTION_TEXTS[nearest_option(np.array([0.9, 0.0, 0.0]))] == "steer full right and coast"
    assert OPTION_TEXTS[nearest_option(np.array([0.0, 0.0, 0.8]))] == "brake"
    frame = np.random.default_rng(0).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    x = preprocess(frame)
    assert x.shape == (3, 48, 48) and x.dtype == torch.uint8
