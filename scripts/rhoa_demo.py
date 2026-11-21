"""
A demo video of Rhoa deciding: real TEST questions (never seen in training), its options, the probabilities,
its pick, and the right answer.

    python scripts/rhoa_demo.py                                   # -> docs/assets/rhoa.mp4 and rhoa.gif
    python scripts/rhoa_demo.py --per_task 2 --out demo.mp4

For each question:   the question appears  ->  the bars grow to Rhoa's probabilities  ->  its pick is marked,
green if right, red if wrong. Questions are picked at random (fixed seed), right or wrong, so the video is honest.
Datasets with many options (CLINC150: 151) show the most likely ones and "... and n more".
"""
import argparse
import os
import random
import textwrap

import torch

from tiny_dlm import video
from tiny_dlm.checkpoint import load_checkpoint
from tiny_dlm.data import collate, load_csv
from tiny_dlm.runtime import resolve_device

TITLES = {"babi": "bAbI · where is someone?", "clinc150": "CLINC150 · which intent? (151 options)",
          "banking77": "Banking77 · which banking intent? (77 options)", "csqa": "CommonsenseQA · common sense"}
W, H, FPS = 960, 540, 25


def draw(task, decision, probs, grow, reveal, shown=8):
    """One frame. grow in [0, 1]: how far the bars have grown. reveal: show the pick and the right answer."""
    from PIL import Image, ImageDraw
    small, font, big = video.load_font(16), video.load_font(20), video.load_font(26)
    img = Image.new("RGB", (W, H), (18, 18, 24))
    d = ImageDraw.Draw(img)
    d.text((30, 20), "Rhoa decides", font=big, fill=(255, 214, 90))
    d.text((W - 30, 26), TITLES.get(task, task), font=small, fill=(160, 160, 175), anchor="ra")
    y = 70
    for line in textwrap.wrap(decision.context, 80)[-4:]:  # long bAbI stories: the last lines matter
        d.text((30, y), line, font=font, fill=(235, 235, 240))
        y += 28
    y += 14
    order = sorted(range(len(decision.candidates)), key=lambda j: -probs[j])[:shown]
    pick = order[0]
    for j in order:
        option, p = decision.candidates[j], float(probs[j])
        right = option in decision.answers
        color = (90, 110, 160)
        if reveal and j == pick:
            color = (80, 200, 120) if right else (220, 80, 80)
        d.rectangle((30, y, 30 + int(320 * p * grow), y + 22), fill=color)
        label = option + ("   ✓ right answer" if reveal and right else "")
        d.text((370, y + 2), label, font=font, fill=(255, 255, 255) if reveal and (j == pick or right) else (170, 170, 185))
        d.text((36 + int(320 * p * grow), y + 2), f"{p:.0%}" if p * grow >= 0.04 else "", font=small, fill=(230, 230, 235))
        y += 32
    if len(decision.candidates) > shown:
        d.text((370, y + 2), f"... and {len(decision.candidates) - shown} more options", font=small, fill=(120, 120, 135))
    if reveal:
        verdict = "Rhoa is right" if decision.candidates[pick] in decision.answers else "Rhoa is wrong"
        d.text((W - 30, H - 40), verdict, font=big, anchor="ra",
               fill=(80, 200, 120) if verdict.endswith("right") else (220, 80, 80))
        if not any(decision.candidates[j] in decision.answers for j in order):  # the answer is not on screen
            d.text((30, H - 36), "right answer: " + " | ".join(decision.answers), font=font, fill=(80, 200, 120))
    return img


def frames_for(task, decision, probs):
    out = [draw(task, decision, probs, 0.0, False)] * int(1.2 * FPS)  # read the question
    out += [draw(task, decision, probs, (n + 1) / 20, False) for n in range(20)]  # bars grow
    out += [draw(task, decision, probs, 1.0, True)] * int(2.2 * FPS)  # the pick
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="checkpoints/rhoa.pt")
    parser.add_argument("--tasks", default="babi,clinc150,banking77,csqa")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--per_task", type=int, default=3)
    parser.add_argument("--out", nargs="+", default=["docs/assets/rhoa.mp4", "docs/assets/rhoa.gif"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    device = resolve_device("auto")
    model, tokenizer, _ = load_checkpoint(args.ckpt, device)
    rng = random.Random(args.seed)
    frames = []
    for task in args.tasks.split(","):
        for decision in rng.sample(load_csv(os.path.join(args.data_root, task, "test.csv")), args.per_task):
            idx, cand_mask, _ = collate([decision], tokenizer, device)
            with torch.no_grad():
                _, probs = model.decide(idx, cand_mask)
            frames += frames_for(task, decision, probs[0].cpu().tolist())
            print(f"{task:10s} {decision.candidates[int(probs[0].argmax())]!r:40s} right: {decision.answers}")

    video.save_video(frames, args.out, fps=FPS, gif_size=(W // 2, H // 2), gif_every=2)

if __name__ == "__main__":
    main()
