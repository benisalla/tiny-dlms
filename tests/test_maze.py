import numpy as np

from tiny_dlm import maze


def corridor():
    """An open board with a wall right of the player: player (0, 0), goal (0, 2), wall (0, 1)."""
    walls = np.zeros((maze.SIZE, maze.SIZE), dtype=bool)
    walls[0, 1] = True
    return walls, (0, 0), (0, 2)


def test_walls_and_borders_block_moves():
    walls, player, _ = corridor()
    assert maze.step(walls, player, "go right") == (0, 0)  # wall
    assert maze.step(walls, player, "go up") == (0, 0)  # border
    assert maze.step(walls, player, "go down") == (1, 0)


def test_teacher_goes_around_the_wall():
    walls, player, goal = corridor()
    assert maze.distances(walls, goal)[player] == 4  # down, right, right, up
    assert maze.best_moves(walls, player, goal) == ["go down"]


def test_picture_has_one_6x6_block_per_cell():
    walls, player, goal = corridor()
    img = maze.render(walls, player, goal)
    assert img.shape == (3, 48, 48)
    assert tuple(img[:, 0, 0]) == maze.COLORS["player"]  # cell (0, 0)
    assert tuple(img[:, 5, 11]) == maze.COLORS["wall"]  # cell (0, 1) covers pixels 6..11
    assert tuple(img[:, 0, 12]) == maze.COLORS["goal"]  # cell (0, 2)
