"""
Let a trained ImageDLM drive CarRacing on tracks it has never seen, and compare it with simple drivers.

    python scripts/car_drive.py                                   # 5 new tracks, every driver
    python scripts/car_drive.py --drivers model --tracks 10
    python scripts/car_drive.py --gif docs/assets/car.gif         # also save a GIF of the model driving

Drivers:
    model     ImageDLM: camera frame -> one of the 15 options (pixels only)
    teacher   the pure-pursuit teacher, rounded to the same 15 options (uses the track waypoints: the upper bound)
    straight  always "go straight and accelerate"
    random    a random option every step

Reward (Gymnasium): +1000/N for every track tile visited, -0.1 per frame. 900+ = "solved".
Test tracks are seeds 1000.. (training used 0.., validation 500..).
"""
import argparse
import random

import numpy as np
import torch

from tiny_dlm.car import OPTION_ACTIONS, OPTION_TEXTS, expert, make_env, nearest_option, preprocess
from tiny_dlm.runtime import resolve_device
from tiny_dlm.vision import encode_options, load_image_dlm


def model_driver(path, device):
    model, tokenizer, option_texts, _ = load_image_dlm(path, device)
    assert option_texts == OPTION_TEXTS, "the checkpoint was trained with other options"
    idx = encode_options(option_texts, tokenizer, device)[None]  # (1, 15, L)
    mask = torch.ones(1, idx.size(1), dtype=torch.bool, device=device)

    def drive(env, obs):
        choice, probs = model.decide(preprocess(obs)[None].to(device), idx, mask)
        return choice.item(), probs[0].cpu()
    return drive


def simple_drivers(rng):
    straight = OPTION_TEXTS.index("go straight and accelerate")
    return {
        "teacher": lambda env, obs: (nearest_option(expert(env)), None),
        "straight": lambda env, obs: (straight, None),
        "random": lambda env, obs: (rng.randrange(len(OPTION_TEXTS)), None),
    }


def run(env, drive, seed, max_steps=1000, frames=None):
    obs, _ = env.reset(seed=seed)
    total = 0.0
    for _ in range(max_steps):
        k, probs = drive(env, obs)
        if frames is not None:
            frames.append((env.render(), k, probs))
        obs, r, terminated, truncated, _ = env.step(OPTION_ACTIONS[k])
        total += r
        if terminated or truncated:
            break
    return total


def save_gif(frames, path, every=2):
    """The rendered frames with the chosen option written on top."""
    import os
    from PIL import Image, ImageDraw
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    images = []
    for frame, k, probs in frames[::every]:
        img = Image.fromarray(frame).resize((384, 384), Image.NEAREST)
        draw = ImageDraw.Draw(img)
        label = OPTION_TEXTS[k] + (f"  ({probs[k]:.0%})" if probs is not None else "")
        draw.rectangle((0, 0, 384, 22), fill=(0, 0, 0))
        draw.text((6, 5), label, fill=(255, 255, 255))
        images.append(img)
    images[0].save(path, save_all=True, append_images=images[1:], duration=40 * every, loop=0, optimize=True)
    print(f"saved {len(images)} frames to {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="checkpoints/car.pt")
    parser.add_argument("--drivers", nargs="+", default=["model", "teacher", "straight", "random"])
    parser.add_argument("--tracks", type=int, default=5)
    parser.add_argument("--first_track", type=int, default=1000)
    parser.add_argument("--gif", default="", help="save a GIF of the model on the first track")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    drivers = simple_drivers(random.Random(args.seed))
    if "model" in args.drivers:
        drivers["model"] = model_driver(args.ckpt, device)
    env = make_env(render_mode="rgb_array" if args.gif else None)

    results = {}
    seeds = range(args.first_track, args.first_track + args.tracks)
    for name in args.drivers:
        rewards = []
        for i, seed in enumerate(seeds):
            frames = [] if (args.gif and name == "model" and i == 0) else None
            rewards.append(run(env, drivers[name], seed, frames=frames))
            if frames:
                save_gif(frames, args.gif)
        results[name] = rewards
        print(f"{name:9s} mean reward {np.mean(rewards):6.0f} | per track " + " ".join(f"{r:.0f}" for r in rewards))
    return results


if __name__ == "__main__":
    main()
