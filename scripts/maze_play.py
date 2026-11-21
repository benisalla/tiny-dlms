"""
Mēlon plays mazes it has never seen, next to simple players, and records a video.

    python scripts/maze_play.py                                        # 200 unseen mazes, every player
    python scripts/maze_play.py --video docs/assets/maze.mp4 docs/assets/maze.gif

Players:
    model     Mēlon: the picture -> one of "go up | go down | go left | go right"
    teacher   breadth-first search: always a shortest path (the upper bound)
    greedy    step toward the goal when that cell is free, else a random free move (no planning)
    random    a random move every step

A player has 40 steps per maze (the longest shortest path is ~25). "Solved" = reached the goal;
"steps / shortest" = how much longer its path was, on the mazes it solved.
"""
import argparse
import random

import torch

from tiny_dlm import maze, video
from tiny_dlm.runtime import resolve_device
from tiny_dlm.vision import encode_options, load_image_dlm


def model_player(path, device):
    model, tokenizer, option_texts, _ = load_image_dlm(path, device)
    assert option_texts == maze.OPTION_TEXTS, "the checkpoint was trained with other options"
    idx = encode_options(option_texts, tokenizer, device)[None]
    mask = torch.ones(1, len(option_texts), dtype=torch.bool, device=device)

    @torch.no_grad()
    def choose(picture, walls, player, goal):
        choice, probs = model.decide(picture[None].to(device), idx, mask)
        return option_texts[choice.item()], probs[0].cpu()
    return choose


def simple_players(rng):
    def greedy(picture, walls, player, goal):
        toward = [m for m in maze.OPTION_TEXTS
                  if abs(maze.step(walls, player, m)[0] - goal[0]) + abs(maze.step(walls, player, m)[1] - goal[1])
                  < abs(player[0] - goal[0]) + abs(player[1] - goal[1])]
        free = [m for m in maze.OPTION_TEXTS if maze.step(walls, player, m) != player]
        return rng.choice(toward or free or maze.OPTION_TEXTS), None
    return {"teacher": lambda pic, w, p, g: (maze.best_moves(w, p, g)[0], None),
            "greedy": greedy,
            "random": lambda pic, w, p, g: (rng.choice(maze.OPTION_TEXTS), None)}


def evaluate(choose, mazes):
    results = [maze.play(choose, w, p, g) + (maze.distances(w, g)[p],) for w, p, g in mazes]
    solved = [(steps, best) for ok, steps, best in results if ok]
    return {"solved": len(solved) / len(results),
            "steps_over_shortest": sum(s for s, _ in solved) / max(1, sum(b for _, b in solved))}


def draw_frame(walls, player, goal, move, probs, n_maze, title="Mēlon plays the maze"):
    """The board on the left (large), the 4 moves with their probabilities on the right."""
    from PIL import Image, ImageDraw
    font, big = video.load_font(20), video.load_font(26)
    img = Image.new("RGB", (900, 450), (18, 18, 24))
    board = maze.render(walls, player, goal).permute(1, 2, 0).numpy()
    img.paste(Image.fromarray(board).resize((432, 432), Image.NEAREST), (9, 9))
    d = ImageDraw.Draw(img)
    d.text((470, 20), title, font=big, fill=(255, 214, 90))
    d.text((470, 56), f"maze {n_maze} · never seen in training", font=font, fill=(160, 160, 175))
    for j, option in enumerate(maze.OPTION_TEXTS):
        y = 120 + j * 56
        p = float(probs[j]) if probs is not None else float(option == move)
        chosen = option == move
        d.rectangle((470, y, 470 + int(220 * p), y + 30), fill=(80, 200, 120) if chosen else (90, 110, 160))
        d.text((705, y + 4), option, font=big, fill=(255, 255, 255) if chosen else (150, 150, 165))
        d.text((476, y + 4), f"{p:.0%}" if p >= 0.08 else "", font=font, fill=(20, 20, 20) if chosen else (230, 230, 235))
    return img


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="checkpoints/melon_maze.pt")
    parser.add_argument("--players", nargs="+", default=["model", "teacher", "greedy", "random"])
    parser.add_argument("--mazes", type=int, default=200)
    parser.add_argument("--video", nargs="*", default=[], help="record the model on 6 mazes: .mp4 and/or .gif")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    players = simple_players(random.Random(0))
    if "model" in args.players or args.video:
        players["model"] = model_player(args.ckpt, resolve_device(args.device))
    mazes = maze.test_mazes(args.mazes)
    results = {}
    for name in args.players:
        results[name] = evaluate(players[name], mazes)
        print(f"{name:8s} solves {results[name]['solved']:6.1%} of {len(mazes)} unseen mazes | "
              f"steps / shortest {results[name]['steps_over_shortest']:.2f}")

    if args.video:
        frames = []
        for n, (w, p, g) in enumerate(mazes[:6], start=1):
            moves = []
            ok, _ = maze.play(players["model"], w, p, g, frames=moves)
            frames += [draw_frame(*m, n) for m in moves]
            last = moves[-1] if moves else (w, p, g, maze.OPTION_TEXTS[0], None)
            end = maze.step(w, last[1], last[3])
            frames += [draw_frame(w, end, g, last[3], last[4], n, "solved!" if ok else "not solved")] * 6
        video.save_video(frames, args.video, fps=4, gif_size=(600, 300))
    return results


if __name__ == "__main__":
    main()
