"""
CarRacing (Gymnasium, Box2D) as a decision problem: from the camera image, pick one option written in words.

    frame (96x96 colour)  ->  ImageDLM  ->  "steer slight left and accelerate": 0.81
                                            "go straight and accelerate":       0.12
                                            ...
                                            "brake":                            0.00

The car wants a continuous action (steer in [-1, 1], gas in [0, 1], brake in [0, 1]).
"How much left or right" becomes the option text: 7 steering amounts x {accelerate, coast} + brake = 15 options.

The teacher (`expert`) drives with the track's waypoints (pure pursuit), which the model never sees.
The model learns to copy it from pixels only (behaviour cloning), then drives on tracks it has never seen.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F

STEERS = [("full left", -1.0), ("medium left", -0.6), ("slight left", -0.25), ("straight", 0.0),
          ("slight right", 0.25), ("medium right", 0.6), ("full right", 1.0)]
GAS = 0.5
BRAKE = 0.8


def _options():
    options = []
    for name, steer in STEERS:
        verb = "go" if name == "straight" else "steer"
        options.append((f"{verb} {name} and accelerate", (steer, GAS, 0.0)))
        options.append((f"{verb} {name} and coast", (steer, 0.0, 0.0)))
    options.append(("brake", (0.0, 0.0, BRAKE)))
    return options


OPTIONS = _options()  # [(text, (steer, gas, brake)), ...]
OPTION_TEXTS = [text for text, _ in OPTIONS]
OPTION_ACTIONS = np.array([action for _, action in OPTIONS], dtype=np.float32)  # (15, 3)

IMAGE_SIZE = 48  # frames are averaged 2x2: 96x96 -> 48x48 (the speed bar at the bottom is kept)
ZOOM_STEPS = 50  # the first ~50 frames are the camera zooming in: not recorded


def make_env(render_mode=None):
    import gymnasium as gym  # only needed for the car: pip install "gymnasium[box2d]"
    return gym.make("CarRacing-v3", render_mode=render_mode)


def preprocess(frame):
    """(96, 96, 3) uint8 frame -> (3, 48, 48) uint8 tensor."""
    x = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float()
    return F.avg_pool2d(x[None], 2)[0].round().to(torch.uint8)


def _signed_angle(forward, target):
    """Angle from `forward` to `target`, positive = target on the left."""
    return math.atan2(forward[0] * target[1] - forward[1] * target[0], forward @ target)


def expert(env, lookahead=6, target_speed=75.0):
    """
        The teacher: pure pursuit on the track's waypoints.
        Aim at the waypoint `lookahead` points ahead, steer toward it, slow down before sharp turns.
        Scores ~920 on average (900 = "solved"). Returns (steer, gas, brake).
    """
    u = env.unwrapped
    car = u.car.hull
    p = np.array(car.position)
    forward = np.array(car.GetWorldVector((0, 1)))
    points = np.array([(t[2], t[3]) for t in u.track])
    i = int(np.argmin(((points - p) ** 2).sum(1)))  # nearest waypoint

    angle = _signed_angle(forward, points[(i + lookahead) % len(points)] - p)
    steer = float(np.clip(-2.0 * angle, -1, 1))  # CarRacing: steer < 0 turns left

    curve = abs(_signed_angle(forward, points[(i + 15) % len(points)] - p))
    speed_goal = target_speed * (0.55 if curve > 0.6 else 1.0)
    speed = float(np.linalg.norm(car.linearVelocity))
    gas = GAS if speed < speed_goal else 0.0
    brake = BRAKE if speed > speed_goal + 10 else 0.0
    return np.array([steer, gas, brake], dtype=np.float32)


def nearest_option(action):
    """A continuous action -> the index of the closest option (what the model must learn to pick)."""
    steer, gas, brake = action
    if brake > 0:
        return len(OPTIONS) - 1
    s = int(np.argmin([abs(steer - value) for _, value in STEERS]))
    return 2 * s + (0 if gas > 0 else 1)
