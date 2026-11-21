"""
A little maze game for the image decision model: from a picture of the maze, pick one move written in words.

    picture (48x48)  ->  Mēlon  ->  "go up": 0.04   "go down": 0.01   "go left": 0.03   "go right": 0.92

    the board: an 8x8 grid, each cell drawn 6 pixels wide (one 6x6 patch per cell)
        wall   dark          player  red
        floor  light         goal    green

The teacher finds the shortest path (breadth-first search). When several moves are equally short, all of them are
right answers (the targets are split between them), so the model is never punished for a good move.
New mazes are generated on the fly: the training data never runs out and a maze is (almost) never seen twice.
"""
import random
from collections import deque

import numpy as np
import torch

SIZE = 8  # cells per side
CELL = 6  # pixels per cell -> 48x48 picture
WALL_PROB = 0.3
MOVES = {"go up": (-1, 0), "go down": (1, 0), "go left": (0, -1), "go right": (0, 1)}
OPTION_TEXTS = list(MOVES)
COLORS = {"wall": (40, 44, 52), "floor": (235, 235, 228), "player": (220, 50, 50), "goal": (40, 170, 90)}


def new_maze(rng, min_distance=4):
    """-> (walls (8, 8) bool, player (row, col), goal (row, col)) with a path between them of >= min_distance steps."""
    while True:
        walls = np.array([[rng.random() < WALL_PROB for _ in range(SIZE)] for _ in range(SIZE)])
        free = [(r, c) for r in range(SIZE) for c in range(SIZE) if not walls[r, c]]
        if len(free) < 2:
            continue
        player, goal = rng.sample(free, 2)
        if distances(walls, goal).get(player, 0) >= min_distance:
            return walls, player, goal


def distances(walls, goal):
    """Breadth-first search from the goal: {cell: number of steps to reach the goal}."""
    dist, queue = {goal: 0}, deque([goal])
    while queue:
        r, c = queue.popleft()
        for dr, dc in MOVES.values():
            n = (r + dr, c + dc)
            if 0 <= n[0] < SIZE and 0 <= n[1] < SIZE and not walls[n] and n not in dist:
                dist[n] = dist[(r, c)] + 1
                queue.append(n)
    return dist


def step(walls, player, move):
    """Move the player; bumping into a wall or the border does nothing."""
    dr, dc = MOVES[move]
    r, c = player[0] + dr, player[1] + dc
    return (r, c) if 0 <= r < SIZE and 0 <= c < SIZE and not walls[r, c] else player


def best_moves(walls, player, goal):
    """The teacher: every move that is on a shortest path to the goal."""
    dist = distances(walls, goal)
    here = dist[player]
    return [m for m in OPTION_TEXTS if dist.get(step(walls, player, m), here) == here - 1]


def render(walls, player, goal):
    """-> (3, 48, 48) uint8 picture, exactly what the network sees."""
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    img[:] = COLORS["floor"]
    img[walls] = COLORS["wall"]
    img[goal] = COLORS["goal"]
    img[player] = COLORS["player"]
    img = img.repeat(CELL, axis=0).repeat(CELL, axis=1)  # each cell -> a 6x6 block
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


def training_example(rng, detour_prob=0.3):
    """
        One (picture, target probabilities) pair. The player is put somewhere along the way to the goal, sometimes
        after a few random moves (detours), so the model also learns to recover from a bad position.
    """
    walls, player, goal = new_maze(rng)
    for _ in range(rng.randint(0, 6)):  # walk part of the way: the teacher's move, or a random one
        if player == goal:
            break
        good = best_moves(walls, player, goal)
        player = step(walls, player, rng.choice(OPTION_TEXTS if rng.random() < detour_prob else good))
    if player == goal:
        return training_example(rng, detour_prob)
    good = best_moves(walls, player, goal)
    targets = torch.tensor([float(m in good) for m in OPTION_TEXTS])
    return render(walls, player, goal), targets / targets.sum()


def batch(batch_size, rng):
    """-> images (b, 3, 48, 48) uint8, targets (b, 4): b new mazes."""
    pairs = [training_example(rng) for _ in range(batch_size)]
    return torch.stack([p for p, _ in pairs]), torch.stack([t for _, t in pairs])


def play(choose, walls, player, goal, max_steps=40, frames=None):
    """
        Let a player `choose(picture) -> (move, probs)` play one maze.
        Returns (reached the goal?, steps taken). max_steps stops a player walking in circles.
    """
    for n in range(max_steps):
        if player == goal:
            return True, n
        move, probs = choose(render(walls, player, goal), walls, player, goal)
        if frames is not None:
            frames.append((walls, player, goal, move, probs))
        player = step(walls, player, move)
    return player == goal, max_steps


def test_mazes(n, seed=1000):
    """The same n unseen mazes for every player (training uses other seeds)."""
    rng = random.Random(seed)
    return [new_maze(rng) for _ in range(n)]
