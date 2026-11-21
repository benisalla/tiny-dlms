"""
The benchmark: every trained model, tested on data never used for training, next to simple baselines.
Writes docs/assets/results.json (read by the project page and scripts/plot_metrics.py).

    python scripts/evaluate_all.py

    Rhoa   test accuracy per dataset with ALL the options, for every Rhoa checkpoint found:
             rhoa (4x128, pre-trained), rhoa_scratch, rhoa_small, rhoa_small_scratch, rhoa_gpt2_scratch
           + per model: tokenizer (word or BPE), vocabulary size, weights, tokens per word on the test text
           baselines: random guessing (mean of 1/k); on bAbI the "last place mentioned" shortcut
    Mēlon  CarRacing score on 5 unseen tracks: melon, melon_small vs teacher, always straight, random
           Maze: share of 200 unseen mazes solved: melon_maze (8x128), melon_maze_4layers vs teacher, greedy, random
"""
import argparse
import json
import os
import random
import sys

import numpy as np

from tiny_dlm.checkpoint import load_checkpoint
from tiny_dlm.data import load_csv
from tiny_dlm.runtime import resolve_device
from tiny_dlm.training import accuracy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the other scripts (car_drive, maze_play)

RHOA = {"rhoa": "Rhoa 4x128 · pre-trained", "rhoa_scratch": "Rhoa 4x128 · no pre-training",
        "rhoa_small": "Rhoa 2x64 · pre-trained", "rhoa_small_scratch": "Rhoa 2x64 · no pre-training",
        "rhoa_gpt2_scratch": "Rhoa 4x128 GPT-2 blocks · no pre-training",
        # the same models with the BPE tokenizer (scripts/compare_tokenizers.sh); the ones above use the word tokenizer
        "rhoa_bpe": "Rhoa 4x128 BPE · pre-trained", "rhoa_bpe_scratch": "Rhoa 4x128 BPE · no pre-training",
        "rhoa_small_bpe": "Rhoa 2x64 BPE · pre-trained", "rhoa_small_bpe_scratch": "Rhoa 2x64 BPE · no pre-training",
        # pre-trained on ~1.5 GB of web text (scripts/web_pretrain.sh), and its no-pre-training twin (same tokenizer)
        "rhoa_web": "Rhoa 4x128 · web pre-training", "rhoa_web_scratch": "Rhoa 4x128 · web tokenizer, no pre-training"}
MELON_CAR = {"melon": "Mēlon 4x128", "melon_small": "Mēlon 2x64"}


def last_place_mentioned(decision):
    """The bAbI shortcut: answer the option that appears last in the story."""
    words = decision.context.split()
    return max(decision.candidates, key=lambda c: max((i for i, w in enumerate(words) if w == c), default=-1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--tasks", default="babi,clinc150,banking77,csqa")
    parser.add_argument("--tracks", type=int, default=5)
    parser.add_argument("--mazes", type=int, default=200)
    parser.add_argument("--max_decisions", type=int, default=0, help="test on the first n decisions only (0 = all)")
    parser.add_argument("--out", default="docs/assets/results.json")
    args = parser.parse_args(argv)
    device = resolve_device("auto")
    found = lambda name: os.path.exists(os.path.join(args.ckpt_dir, f"{name}.pt"))  # noqa: E731
    ckpt = lambda name: os.path.join(args.ckpt_dir, f"{name}.pt")  # noqa: E731

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    results["names"] = {**RHOA, **MELON_CAR, "melon_maze": "Mēlon 8x128", "melon_maze_4layers": "Mēlon 4x128"}

    # ---------------------------------------------------------------- Rhoa
    results["test"] = {}
    models = {name: load_checkpoint(ckpt(name), device)[:2] for name in RHOA if found(name)}
    test_texts = [t for task in args.tasks.split(",")
                  for d in load_csv(os.path.join(args.data_root, task, "test.csv"))[:500] for t in [d.context, *d.candidates[:3]]]
    words = sum(len(t.split()) for t in test_texts)
    results["tokenizers"] = {name: {"tokenizer": "bpe" if type(tok).__name__ == "BPETokenizer" else "word",
                                    "vocab": len(tok), "weights": m.get_num_params(),
                                    "tokens_per_word": sum(len(tok.encode(t)) for t in test_texts) / words,
                                    "unknown": sum(tok.encode(t).count(tok.unk_id) for t in test_texts) / words}
                             for name, (m, tok) in models.items()}
    for name, row in results["tokenizers"].items():
        print(f"{name:24s} {row['tokenizer']:4s} vocab {row['vocab']:6,} | {row['weights'] / 1e6:.2f} M weights | "
              f"{row['tokens_per_word']:.2f} tokens/word | {row['unknown']:.2%} unknown")
    for task in args.tasks.split(","):
        test = load_csv(os.path.join(args.data_root, task, "test.csv"))
        test = test[:args.max_decisions] if args.max_decisions else test
        row = {"options": max(len(d.candidates) for d in test), "decisions": len(test),
               "baselines": {"random": float(np.mean([len(d.answers) / len(d.candidates) for d in test]))},
               "models": {name: accuracy(m, test, tok, device) for name, (m, tok) in models.items()}}
        if task == "babi":
            row["baselines"]["last place mentioned"] = float(np.mean([last_place_mentioned(d) in d.answers for d in test]))
        results["test"][task] = row
        print(f"{task:10s} " + " | ".join(f"{k} {v:.1%}" for k, v in {**row["baselines"], **row["models"]}.items()))

    # ---------------------------------------------------------------- Mēlon drives
    if any(found(n) for n in MELON_CAR):
        from car_drive import model_driver, run, simple_drivers
        from tiny_dlm.car import make_env
        env = make_env()
        players = {**simple_drivers(random.Random(0)), **{n: model_driver(ckpt(n), device) for n in MELON_CAR if found(n)}}
        seeds = range(1000, 1000 + args.tracks)
        scores = {name: [run(env, play, seed) for seed in seeds] for name, play in players.items()}
        results["car"] = {name: {"mean": float(np.mean(v)), "per_track": v} for name, v in scores.items()}
        print("car (unseen tracks): " + " | ".join(f"{k} {np.mean(v):.0f}" for k, v in scores.items()))

    # ---------------------------------------------------------------- Mēlon plays the maze
    if found("melon_maze"):
        from maze_play import evaluate, model_player, simple_players
        from tiny_dlm import maze
        mazes = maze.test_mazes(args.mazes)
        players = {**simple_players(random.Random(0)),
                   **{n: model_player(ckpt(n), device) for n in ("melon_maze_4layers", "melon_maze") if found(n)}}
        results["maze"] = {name: evaluate(play, mazes) for name, play in players.items()}
        print("maze (unseen): " + " | ".join(f"{k} {v['solved']:.0%}" for k, v in results["maze"].items()))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
