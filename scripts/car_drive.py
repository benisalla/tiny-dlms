"""
Let a trained ImageDLM drive CarRacing on tracks it has never seen, and compare it with simple drivers.

    python scripts/car_drive.py                                   # 5 new tracks, every driver
    python scripts/car_drive.py --drivers model --tracks 10
    python scripts/car_drive.py --video docs/assets/melon.mp4     # also record the model driving (.mp4 or .gif)

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

from tiny_dlm import video
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
        j, probs = drive(env, obs)  # j: the chosen option
        if frames is not None:
            frames.append((env.render(), j, probs, total))
        obs, r, terminated, truncated, _ = env.step(OPTION_ACTIONS[j])
        total += r
        if terminated or truncated:
            break
    return total


def draw_frame(frame, j, probs, score, title="Mēlon drives"):
    """
        One video frame: the car on the left, every option with its probability on the right.

            ┌──────────────┬──────────────────────────────────────┐
            │              │ Mēlon drives            score 412    │
            │   the road   │ steer full left and accelerate   ▏ 0%│
            │   (camera)   │ steer slight left and accelerate ███ │  <- chosen option in green
            │              │ ...                     15 options   │
            └──────────────┴──────────────────────────────────────┘
    """
    from PIL import Image, ImageDraw
    font, big = video.load_font(17), video.load_font(24)
    if not big.getmask("ē").getbbox():  # the built-in font has no "ē"
        title = title.replace("ē", "e")
    img = Image.new("RGB", (900, 450), (18, 18, 24))
    img.paste(Image.fromarray(frame).resize((450, 450), Image.NEAREST), (0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((470, 14), title, font=big, fill=(255, 214, 90))
    draw.text((760, 18), f"score {score:4.0f}", font=font, fill=(200, 200, 210))
    for n, text in enumerate(OPTION_TEXTS):
        y = 58 + n * 25
        p = float(probs[n]) if probs is not None else float(n == j)
        chosen = n == j
        draw.rectangle((470, y, 470 + int(120 * p), y + 20), fill=(80, 200, 120) if chosen else (90, 110, 160))
        draw.text((600, y), text, font=font, fill=(255, 255, 255) if chosen else (150, 150, 165))
        if probs is not None and p >= 0.05:  # inside a long bar, just after a short one: never on the option names
            draw.text((474, y) if p >= 0.4 else (474 + int(120 * p), y), f"{p:.0%}", font=font,
                      fill=(18, 18, 24) if p >= 0.4 else (220, 220, 230))
    return img


def save_video(frames, paths):
    """One video frame for every second step; the GIF is a light ~24 s version for READMEs."""
    video.save_video([draw_frame(*f) for f in frames[::2]], paths, fps=25, gif_size=(480, 240), gif_every=2, gif_max=150)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="checkpoints/melon.pt")
    parser.add_argument("--drivers", nargs="+", default=["model", "teacher", "straight", "random"])
    parser.add_argument("--tracks", type=int, default=5)
    parser.add_argument("--first_track", type=int, default=1000)
    parser.add_argument("--video", nargs="*", default=[], help="record the model on the first track: .mp4 and/or .gif")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    drivers = simple_drivers(random.Random(args.seed))
    if "model" in args.drivers:
        drivers["model"] = model_driver(args.ckpt, device)
    env = make_env(render_mode="rgb_array" if args.video else None)

    results = {}
    seeds = range(args.first_track, args.first_track + args.tracks)
    for name in args.drivers:
        rewards = []
        for i, seed in enumerate(seeds):
            frames = [] if (args.video and name == "model" and i == 0) else None
            rewards.append(run(env, drivers[name], seed, frames=frames))
            if frames:
                save_video(frames, args.video)
        results[name] = rewards
        print(f"{name:9s} mean reward {np.mean(rewards):6.0f} | per track " + " ".join(f"{r:.0f}" for r in rewards))
    return results


if __name__ == "__main__":
    main()
