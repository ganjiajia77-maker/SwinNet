import math

import torch


CONNECTIVITY_DIR_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
CONNECTIVITY_DIRECTIONS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)
CONNECTIVITY_OPPOSITE = (4, 5, 6, 7, 0, 1, 2, 3)

AXIAL_DIR_NAMES = ("N/S", "NE/SW", "E/W", "SE/NW")
AXIAL_DIRECTIONS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


def axial_double_angle_basis(directions=AXIAL_DIRECTIONS, *, dtype=torch.float32):
    basis = []
    for dy, dx in directions:
        theta = math.atan2(float(dy), float(dx))
        basis.append([math.cos(2.0 * theta), math.sin(2.0 * theta)])
    return torch.tensor(basis, dtype=dtype)


def connectivity_double_angle_basis(*, dtype=torch.float32):
    return axial_double_angle_basis(CONNECTIVITY_DIRECTIONS, dtype=dtype)

