"""
Record the teacher driving CarRacing: (frame, the option the teacher picks) pairs, for behaviour cloning.

    python scripts/car_record.py                          # 40 train tracks + 5 val tracks -> data/car/{train,val}.pt
    python scripts/car_record.py --train_tracks 10        # quicker

Why noise? A student that only ever sees perfect driving never sees the car drifting off the road,
so when it makes its first small mistake it is lost. Here the car sometimes executes a WRONG action
(noisy steering, or a few steps steering a notch too far), but every frame is still labelled with what the teacher
WOULD do there. The student learns how to come back to the road (DART, Laskey et al. 2017).

Tracks: train = seeds 0.., val = seeds 500.., and scripts/car_drive.py tests on seeds 1000.. (never recorded).

    file             tensor     dtype   shape
    ================================================
    images           uint8      (n, 3, 48, 48)   n frames
    labels           long       (n,)    index into tiny_dlm.car.OPTION_TEXTS
    option_texts     list[str]  15
"""
import argparse
import os
import random
import time

import numpy as np
import torch

from tiny_dlm.car import OPTION_ACTIONS, OPTION_TEXTS, ZOOM_STEPS, expert, make_env, nearest_option, preprocess


def record(env, seeds, steer_noise, drift_prob, rng, max_steps=1000):
    images, labels, rewards = [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        total, drift = 0.0, 0
        for t in range(max_steps):
            teacher = nearest_option(expert(env))
            if t >= ZOOM_STEPS:
                images.append(preprocess(obs))
                labels.append(teacher)
            # what the car actually does: the teacher, sometimes perturbed
            action = OPTION_ACTIONS[teacher].copy()
            if drift == 0 and rng.random() < drift_prob:
                drift = rng.randint(3, 8)  # a short wrong stretch: steer one notch too far left or right
                drift_shift = rng.choice([-0.35, 0.35])
            if drift > 0:
                action[0] += drift_shift
                drift -= 1
            action[0] = np.clip(action[0] + rng.gauss(0, steer_noise), -1, 1)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            if terminated or truncated:
                break
        rewards.append(total)
        print(f"  track {seed}: {t + 1} steps, reward {total:.0f} (with noise)")
    return torch.stack(images), torch.tensor(labels), rewards


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_tracks", type=int, default=40)
    parser.add_argument("--val_tracks", type=int, default=5)
    parser.add_argument("--steer_noise", type=float, default=0.1)
    parser.add_argument("--drift_prob", type=float, default=0.02, help="chance per step to start a wrong stretch")
    parser.add_argument("--out_dir", default="data/car")
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    env, rng = make_env(), random.Random(args.seed)
    for split, seeds in (("train", range(args.train_tracks)), ("val", range(500, 500 + args.val_tracks))):
        t0 = time.time()
        print(f"{split}:")
        images, labels, rewards = record(env, seeds, args.steer_noise, args.drift_prob, rng)
        path = os.path.join(args.out_dir, f"{split}.pt")
        torch.save({"images": images, "labels": labels, "option_texts": OPTION_TEXTS}, path)
        counts = torch.bincount(labels, minlength=len(OPTION_TEXTS))
        print(f"{split}: {len(labels):,} frames from {len(seeds)} tracks, mean reward {np.mean(rewards):.0f}, "
              f"{images.numel() / 1e6:.0f} MB, {time.time() - t0:.0f}s -> {path}")
        print("  most common: " + ", ".join(f"{OPTION_TEXTS[i]} {counts[i] / len(labels):.0%}"
                                            for i in counts.argsort(descending=True)[:3]))


if __name__ == "__main__":
    main()
