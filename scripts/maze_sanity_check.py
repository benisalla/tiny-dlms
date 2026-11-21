"""
Health checks of the maze game and of Mēlon playing it, run BEFORE training (seconds).

    python scripts/maze_sanity_check.py --n_layer 4 --n_embd 128

    game      1. the teacher solves 100 mazes by the shortest path; targets sum to 1
              2. look at the data: 16 pictures with their right moves -> checkpoints/maze_batch.png
    model     3. loss at init = random guessing (log 4)
              4. every parameter gets a finite, non-zero gradient
              5. activations stay healthy from block to block
              6. one optimizer step moves the weights a sane amount
              7. shuffling the options only shuffles the probabilities
              8. the same seed gives the same loss
    learning  9. it memorises 32 mazes
"""
import argparse
import math
import os
import random

import torch

from tiny_dlm import maze, sanity
from tiny_dlm.config import ModelConfig
from tiny_dlm.sanity import check
from tiny_dlm.tokenizer import WordTokenizer
from tiny_dlm.vision import ImageDLM, encode_options


def save_batch_picture(images, targets, path):
    from PIL import Image, ImageDraw
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    grid = Image.new("RGB", (4 * 180, 4 * 196), "white")  # 168 px boards with white gaps between them
    for n, (img, t) in enumerate(zip(images[:16], targets[:16])):
        tile = Image.fromarray(img.permute(1, 2, 0).numpy()).resize((168, 168), Image.NEAREST)
        grid.paste(tile, ((n % 4) * 180, (n // 4) * 196))
        right = " / ".join(m.replace("go ", "") for m, p in zip(maze.OPTION_TEXTS, t) if p > 0)
        ImageDraw.Draw(grid).text(((n % 4) * 180 + 4, (n // 4) * 196 + 172), right, fill="black")
    grid.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--picture", default="checkpoints/maze_batch.png")
    args = parser.parse_args(argv)
    torch.manual_seed(0)
    sanity.RESULTS.clear()

    print("game")
    teacher = lambda img, w, p, g: (maze.best_moves(w, p, g)[0], None)  # noqa: E731
    ok = all(maze.play(teacher, w, p, g) == (True, maze.distances(w, g)[p]) for w, p, g in maze.test_mazes(100))
    images, targets = maze.batch(64, random.Random(0))
    check("1. the teacher solves 100 mazes by the shortest path; targets sum to 1",
          ok and torch.allclose(targets.sum(1), torch.ones(64)))
    save_batch_picture(images, targets, args.picture)
    print(f"  [LOOK] 2. 16 pictures with their right moves, exactly as the network gets them: {args.picture}")

    print("model")
    k = len(maze.OPTION_TEXTS)
    tok = WordTokenizer.from_texts(maze.OPTION_TEXTS)
    cfg = ModelConfig(vocab_size=len(tok), block_size=72, n_layer=args.n_layer, n_embd=args.n_embd,
                      n_head=args.n_head, drop_rate=0.0, pad_id=tok.pad_id, sep_id=tok.sep_id)
    build = lambda: ImageDLM(cfg, image_size=48, patch_size=maze.CELL)  # noqa: E731
    ids, mask = encode_options(maze.OPTION_TEXTS, tok).expand(64, -1, -1), torch.ones(64, k, dtype=torch.bool)
    run = lambda m: m(images, ids, mask, targets)  # noqa: E731
    model = build()
    sanity.check_init_loss("3. loss at init = random guessing", run(model)[1].item(), math.log(k))
    sanity.check_gradients("4. every parameter gets a gradient", model, run)
    sanity.check_activations("5. activations stay healthy", model, model.transformer.h, run)
    sanity.check_update_ratio("6. one optimizer step moves the weights a sane amount", model, run)
    model.eval()
    with torch.no_grad():
        perm = torch.tensor([2, 0, 3, 1])
        p = torch.softmax(model(images[:1], ids[:1], mask[:1])[0], -1)[0]
        q = torch.softmax(model(images[:1], ids[:1, perm], mask[:1])[0], -1)[0]
    check("7. shuffling the options only shuffles the probabilities", torch.allclose(p[perm], q, atol=1e-5))
    sanity.check_reproducible("8. the same seed gives the same loss", build, run)

    print("learning")
    model = build()
    small = lambda m: m(images[:32], ids[:32], mask[:32], targets[:32])  # noqa: E731
    floor = -(targets[:32] * targets[:32].clamp(min=1e-12).log()).sum(-1).mean().item()  # > 0: ties between moves
    loss, steps = sanity.overfit(model, small, steps=500, lr=1e-3, target=floor + 0.02)
    model.eval()
    with torch.no_grad():
        choice = small(model)[0].argmax(-1)
    acc = (targets[torch.arange(32), choice] > 0).float().mean().item()
    check("9. memorise 32 mazes", acc == 1.0, f"loss {loss:.3f} (best possible {floor:.3f}) after {steps} steps, "
          f"{acc:.0%} good moves")
    return sanity.summary()


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
