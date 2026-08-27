import math

import torch

from topology_direction_constants import (
    AXIAL_DIR_NAMES,
    AXIAL_DIRECTIONS,
    CONNECTIVITY_DIR_NAMES,
    CONNECTIVITY_DIRECTIONS,
    CONNECTIVITY_OPPOSITE,
    axial_double_angle_basis,
    connectivity_double_angle_basis,
)


def fmt_vec(vec):
    return "(" + ", ".join(f"{float(x):+.3f}" for x in vec) + ")"


def main():
    print("Direction representation")
    print("  direction_gt channels: [cos(2*theta), sin(2*theta)]")
    print("  direction loss target: continuous axial vector, cosine loss on skeleton pixels")
    print("  direction head channels: 2 continuous channels, not class logits")

    print("\n4-axis evaluation mapping")
    axial_basis = axial_double_angle_basis()
    for idx, (name, offset, basis) in enumerate(zip(AXIAL_DIR_NAMES, AXIAL_DIRECTIONS, axial_basis)):
        dy, dx = offset
        theta = math.atan2(float(dy), float(dx))
        print(
            f"  AXIS {idx}: {name:7s} offset=({dy:+d},{dx:+d}) "
            f"theta={math.degrees(theta):+7.2f}deg double_angle_basis={fmt_vec(basis)}"
        )

    print("\n8-dir connectivity mapping")
    conn_basis = connectivity_double_angle_basis()
    for idx, (name, offset, basis) in enumerate(zip(CONNECTIVITY_DIR_NAMES, CONNECTIVITY_DIRECTIONS, conn_basis)):
        dy, dx = offset
        opposite = CONNECTIVITY_OPPOSITE[idx]
        print(
            f"  CONN {idx}: {name:2s} offset=({dy:+d},{dx:+d}) "
            f"opposite={opposite}:{CONNECTIVITY_DIR_NAMES[opposite]:2s} "
            f"dir_align_basis={fmt_vec(basis)}"
        )

    expected_opposite = (4, 5, 6, 7, 0, 1, 2, 3)
    if tuple(CONNECTIVITY_OPPOSITE) != expected_opposite:
        raise AssertionError(f"Bad opposite mapping: {CONNECTIVITY_OPPOSITE}")

    for idx, opposite in enumerate(CONNECTIVITY_OPPOSITE):
        if not torch.allclose(conn_basis[idx], conn_basis[opposite], atol=1e-6):
            raise AssertionError(
                f"Opposite directions must share axial double-angle basis: "
                f"{CONNECTIVITY_DIR_NAMES[idx]} vs {CONNECTIVITY_DIR_NAMES[opposite]}"
            )

    for axial_idx, conn_idx in enumerate((0, 1, 2, 3)):
        if not torch.allclose(axial_basis[axial_idx], conn_basis[conn_idx], atol=1e-6):
            raise AssertionError(
                f"4-axis mapping {AXIAL_DIR_NAMES[axial_idx]} does not match "
                f"connectivity basis {CONNECTIVITY_DIR_NAMES[conn_idx]}"
            )

    print("\nOK: direction target/loss/eval basis, dir_align_d basis, connectivity offsets, and reciprocal mapping are consistent.")


if __name__ == "__main__":
    main()

