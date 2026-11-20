"""
The final test of every trained model, on data never used for training. Writes docs/assets/results.json.

    python scripts/evaluate_all.py

    Rhoa   test accuracy per dataset with ALL the options: with pre-training (rhoa.pt) vs without (rhoa_scratch.pt),
           next to random guessing (the mean of 1/k)
    Mēlon  CarRacing score on tracks never seen (seeds 1000+): Mēlon vs the teacher vs random
"""
import argparse
import json
import os
import random

import numpy as np

from tiny_dlm.checkpoint import load_checkpoint
from tiny_dlm.data import load_csv
from tiny_dlm.runtime import resolve_device
from tiny_dlm.training import accuracy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--tasks", default="babi,clinc150,banking77,csqa")
    parser.add_argument("--tracks", type=int, default=5)
    parser.add_argument("--max_decisions", type=int, default=0, help="test on the first n decisions only (0 = all)")
    parser.add_argument("--out", default="docs/assets/results.json")
    args = parser.parse_args(argv)
    device = resolve_device("auto")

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    results["test"] = {}
    models = {name: load_checkpoint(os.path.join(args.ckpt_dir, f"{file}.pt"), device)[:2]
              for name, file in (("with_pretraining", "rhoa"), ("without_pretraining", "rhoa_scratch"))
              if os.path.exists(os.path.join(args.ckpt_dir, f"{file}.pt"))}
    for task in args.tasks.split(","):
        test = load_csv(os.path.join(args.data_root, task, "test.csv"))
        test = test[:args.max_decisions] if args.max_decisions else test
        row = {"options": max(len(d.candidates) for d in test), "decisions": len(test),
               "random": float(np.mean([len(d.answers) / len(d.candidates) for d in test]))}
        for name, (model, tokenizer) in models.items():
            row[name] = accuracy(model, test, tokenizer, device)
        results["test"][task] = row
        print(f"{task:10s} " + " | ".join(f"{k} {v:.1%}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))

    melon = os.path.join(args.ckpt_dir, "melon.pt")
    if os.path.exists(melon):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from car_drive import model_driver, run, simple_drivers
        from tiny_dlm.car import make_env
        env, drivers = make_env(), simple_drivers(random.Random(0))
        drivers["model"] = model_driver(melon, device)
        seeds = range(1000, 1000 + args.tracks)
        scores = {name: [run(env, drivers[name], seed) for seed in seeds] for name in ("model", "teacher", "random")}
        results["melon_drive"] = {name: float(np.mean(v)) for name, v in scores.items()}
        results["melon_drive"]["per_track"] = scores
        print("Mēlon on unseen tracks: " + " | ".join(f"{k} {np.mean(v):.0f}" for k, v in scores.items()))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
