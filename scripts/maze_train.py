"""
Train Mēlon (ImageDLM) to play the maze from pictures: pick "go up | go down | go left | go right".

    python scripts/maze_train.py                     # -> checkpoints/melon_maze.pt
    python scripts/maze_play.py                      # play unseen mazes, compare with simple players, video

New mazes are generated for every batch (tiny_dlm/maze.py), so there is no dataset to prepare.
The picture is 48x48 with one 6x6 patch per cell (64 patches). The targets are split between every move on a
shortest path. Every evaluation also lets the model PLAY 50 unseen mazes: "solved" is the share it finishes.
"""
import os
import random

import torch

from tiny_dlm import maze
from tiny_dlm.config import ModelConfig, TrainConfig
from tiny_dlm.runtime import parse_config_args, resolve_device, setup_torch
from tiny_dlm.tokenizer import build_tokenizer
from tiny_dlm.training import build_optimizer, train
from tiny_dlm.vision import ImageDLM, encode_options

DEFAULTS = dict(ckpt_name="melon_maze.pt", batch_size=64, max_iters=4000, lr_decay_iters=4000, warmup_iters=100,
                eval_interval=250, eval_iters=10, log_interval=100,
                block_size=72)  # 64 patches + "[SEP] go right"
PATCH = maze.CELL  # one patch per cell


def model_player(model, option_idx, device):
    """A maze player that asks the model: picture -> (move, probabilities)."""
    mask = torch.ones(1, len(maze.OPTION_TEXTS), dtype=torch.bool, device=device)

    @torch.no_grad()
    def choose(picture, walls, player, goal):
        choice, probs = model.decide(picture[None].to(device), option_idx[None], mask)
        return maze.OPTION_TEXTS[choice.item()], probs[0].cpu()
    return choose


def solve_rate(model, option_idx, device, mazes):
    """Share of mazes finished, and steps taken / shortest path on the finished ones."""
    model.eval()
    choose = model_player(model, option_idx, device)
    results = [maze.play(choose, w, p, g) + (maze.distances(w, g)[p],) for w, p, g in mazes]
    model.train()
    solved = [(steps, best) for ok, steps, best in results if ok]
    extra = sum(s for s, _ in solved) / max(1, sum(b for _, b in solved))
    return len(solved) / len(results), extra


def main(argv=None):
    cfg, model_cfg = parse_config_args(TrainConfig, ModelConfig, description=__doc__, argv=argv, defaults=DEFAULTS)
    setup_torch(cfg.seed)
    device = resolve_device(cfg.device)

    tokenizer = build_tokenizer(cfg.tokenizer, maze.OPTION_TEXTS, cfg.bpe_vocab_size)
    option_idx = encode_options(maze.OPTION_TEXTS, tokenizer, device)  # (k = 4 options, s tokens)
    model_cfg.vocab_size, model_cfg.pad_id, model_cfg.sep_id = len(tokenizer), tokenizer.pad_id, tokenizer.sep_id
    model = ImageDLM(model_cfg, image_size=maze.SIZE * maze.CELL, patch_size=PATCH).to(device)
    print(f"Mēlon (maze) | number of parameters: {model.get_num_params() / 1e3:.1f}K | {len(maze.OPTION_TEXTS)} options "
          f"| device: {device}")

    k = len(maze.OPTION_TEXTS)
    train_rng = random.Random(cfg.seed)
    val_images, val_targets = maze.batch(512, random.Random(500))  # fixed validation pictures

    def train_batch():
        images, targets = maze.batch(cfg.batch_size, train_rng)
        b = len(images)
        return (images.to(device), option_idx.expand(b, -1, -1), torch.ones(b, k, dtype=torch.bool, device=device),
                targets.to(device))

    def val_batch():
        rows = torch.randint(0, len(val_images), (cfg.batch_size,))
        b = len(rows)
        return (val_images[rows].to(device), option_idx.expand(b, -1, -1),
                torch.ones(b, k, dtype=torch.bool, device=device), val_targets[rows].to(device))

    @torch.no_grad()
    def good_move_rate(m):
        """How often the chosen move is one of the shortest-path moves (fixed validation pictures)."""
        m.eval()
        b = len(val_images)
        choice, _ = m.decide(val_images.to(device), option_idx.expand(b, -1, -1),
                             torch.ones(b, k, dtype=torch.bool, device=device))
        m.train()
        return (val_targets[torch.arange(b), choice.cpu()] > 0).float().mean().item()

    eval_mazes = maze.test_mazes(50, seed=700)
    eval_fn = lambda m: {"val_acc/good_move": good_move_rate(m),  # noqa: E731
                         "val_acc/solved": solve_rate(m, option_idx, device, eval_mazes)[0]}
    best = train(model, build_optimizer(model, cfg, device), cfg, {"train": train_batch, "val": val_batch},
                 eval_fn=eval_fn, ckpt_extra={"stage": "maze", "tokenizer": tokenizer.to_dict(),
                                              "option_texts": maze.OPTION_TEXTS, "image_args": model.image_args})
    rate, extra = solve_rate(model, option_idx, device, maze.test_mazes(200))
    print(f"done, best val loss {best:.4f} | good moves {good_move_rate(model):.1%} | solves {rate:.1%} of 200 unseen "
          f"mazes, {extra:.2f}x the shortest path -> {os.path.join(cfg.out_dir, cfg.ckpt_name)}")
    return {"best_val_loss": best, "solved": rate}


if __name__ == "__main__":
    main()
