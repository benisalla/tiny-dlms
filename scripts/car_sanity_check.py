"""
Health checks of the car data and of ImageDLM, run BEFORE training (seconds).

    python scripts/car_sanity_check.py
    python scripts/car_sanity_check.py --n_layer 4 --n_embd 128     # the size you plan to train

    data      1. frames and labels look right (shapes, pixel range, every option is used)
              2. train and val come from different tracks (no frame is shared)
              3. look at the data: 16 frames with their labels -> checkpoints/car_batch.png
    model     4. loss at init = random guessing (log 15)
              5. every parameter gets a finite, non-zero gradient
              6. activations stay healthy from block to block
              7. one optimizer step moves the weights by a sane amount
              8. images in a batch do not leak into each other
              9. shuffling the options only shuffles the probabilities
             10. the model looks at the image (different frames -> different scores)
             11. the same seed gives the same loss
    learning 12. it memorises 32 real frames (if it cannot, nothing else will work)
"""
import argparse
import math
import os

import torch

from tiny_dlm import sanity
from tiny_dlm.config import ModelConfig
from tiny_dlm.sanity import check
from tiny_dlm.tokenizer import WordTokenizer
from tiny_dlm.vision import ImageDLM, encode_options


def save_batch_picture(images, labels, texts, path):
    """A 4x4 grid of frames exactly as the network gets them (48x48), each with its label."""
    from PIL import Image, ImageDraw
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    grid = Image.new("RGB", (4 * 192, 4 * 210), "white")
    for n, (img, y) in enumerate(zip(images[:16], labels[:16])):
        tile = Image.fromarray(img.permute(1, 2, 0).numpy()).resize((192, 192), Image.NEAREST)
        grid.paste(tile, ((n % 4) * 192, (n // 4) * 210))
        ImageDraw.Draw(grid).text(((n % 4) * 192 + 4, (n // 4) * 210 + 194), texts[y], fill="black")
    grid.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data/car")
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--picture", default="checkpoints/car_batch.png")
    args = parser.parse_args(argv)
    torch.manual_seed(0)

    train = torch.load(os.path.join(args.data_dir, "train.pt"), weights_only=True)
    val = torch.load(os.path.join(args.data_dir, "val.pt"), weights_only=True)
    texts, imgs, labels = train["option_texts"], train["images"], train["labels"]
    K = len(texts)

    print("data")
    used = torch.bincount(labels, minlength=K)
    check("1. frames and labels look right",
          imgs.shape[1:] == (3, 48, 48) and imgs.dtype == torch.uint8 and imgs.float().std() > 10
          and labels.min() >= 0 and labels.max() < K and bool((used > 0).all()),
          f"{len(labels):,} frames, pixel std {imgs.float().std():.0f}, rarest option used {used.min()} times")
    sample = val["images"][::50]
    shared = sum(bool((imgs[::10] == v).flatten(1).all(1).any()) for v in sample)
    check("2. train and val come from different tracks", shared == 0, f"{shared} of {len(sample)} val frames in train")
    i = torch.randperm(len(labels))[:64]
    x, y = imgs[i], labels[i]
    save_batch_picture(x, y, texts, args.picture)
    print(f"  [LOOK] 3. 16 frames with their labels, exactly as the network gets them: {args.picture}")

    print("model")
    tok = WordTokenizer.from_texts(texts)
    cfg = ModelConfig(vocab_size=len(tok), block_size=48, n_layer=args.n_layer, n_embd=args.n_embd, n_head=4,
                      drop_rate=0.0, pad_id=tok.pad_id, sep_id=tok.sep_id)  # no dropout: checks are deterministic
    build = lambda: ImageDLM(cfg)  # noqa: E731
    ids, mask = encode_options(texts, tok).expand(64, -1, -1), torch.ones(64, K, dtype=torch.bool)
    targets = torch.nn.functional.one_hot(y, K).float()
    run = lambda m: m(x, ids, mask, targets)  # noqa: E731
    model = build()

    sanity.check_init_loss("4. loss at init = random guessing", run(model)[1].item(), math.log(K))
    sanity.check_gradients("5. every parameter gets a gradient", model, run)
    sanity.check_activations("6. activations stay healthy", model, model.transformer.h, run)
    sanity.check_update_ratio("7. one optimizer step moves the weights a sane amount", model, run)

    model.eval()
    with torch.no_grad():
        full, _ = model(x[:4], ids[:4], mask[:4])
        changed = x[:4].clone()
        changed[1:] = 0
        alone, _ = model(changed, ids[:4], mask[:4])
        check("8. images in a batch do not leak into each other", torch.allclose(full[0], alone[0], atol=1e-5),
              f"max difference {(full[0] - alone[0]).abs().max():.1e}")
        perm = torch.randperm(K)
        p = torch.softmax(model(x[:1], ids[:1], mask[:1])[0], -1)[0]
        q = torch.softmax(model(x[:1], ids[:1, perm], mask[:1])[0], -1)[0]
        check("9. shuffling the options only shuffles the probabilities", torch.allclose(p[perm], q, atol=1e-5))
        spread = (full[0] - full[1]).abs().max().item()
        check("10. the model looks at the image", spread > 1e-4, f"two frames differ by up to {spread:.1e} in logit")
    sanity.check_reproducible("11. the same seed gives the same loss", build, run)

    print("learning")
    model = build()
    small = lambda m: m(x[:32], ids[:32], mask[:32], targets[:32])  # noqa: E731
    loss, steps = sanity.overfit(model, small, steps=500, lr=1e-3, target=0.01)
    model.eval()
    with torch.no_grad():
        acc = (small(model)[0].argmax(-1) == y[:32]).float().mean().item()
    check("12. memorise 32 real frames", acc == 1.0, f"loss {loss:.3f} after {steps} steps, {acc:.0%} correct")
    return sanity.summary()


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
