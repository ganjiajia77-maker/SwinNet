import torch

from losses.road_losses import build_connectivity_target
from networks.skeleton_guided_head import PairwiseConnectivityHead
from topology_direction_constants import CONNECTIVITY_DIR_NAMES, CONNECTIVITY_DIRECTIONS


def main():
    height = 5
    width = 5
    feature = torch.arange(height * width, dtype=torch.float32).view(1, 1, height, width)
    skeleton = torch.zeros(1, 1, height, width)
    center_y = 2
    center_x = 2
    skeleton[0, 0, center_y, center_x] = 1.0
    for dy, dx in CONNECTIVITY_DIRECTIONS:
        skeleton[0, 0, center_y + dy, center_x + dx] = 1.0

    target = build_connectivity_target(skeleton)
    print("Connectivity pair alignment audit")
    print(f"  center p=({center_y},{center_x}) feature={feature[0, 0, center_y, center_x].item():.0f}")
    mismatches = 0
    for idx, (name, (dy, dx)) in enumerate(zip(CONNECTIVITY_DIR_NAMES, CONNECTIVITY_DIRECTIONS)):
        shifted = PairwiseConnectivityHead._shift_feature(feature, dy, dx)
        actual_neighbor = int(shifted[0, 0, center_y, center_x].item())
        expected_neighbor = int(feature[0, 0, center_y + dy, center_x + dx].item())
        target_value = int(target[0, idx, center_y, center_x].item())
        ok = actual_neighbor == expected_neighbor and target_value == 1
        if not ok:
            mismatches += 1
        print(
            f"  {idx}:{name:2s} offset=({dy:+d},{dx:+d}) "
            f"target={target_value} expected_q=({center_y + dy},{center_x + dx}) "
            f"expected_feat={expected_neighbor} actual_shift_feat={actual_neighbor} "
            f"{'OK' if ok else 'MISMATCH'}"
        )

    if mismatches:
        raise AssertionError(f"Connectivity feature-pair alignment has {mismatches} mismatches.")
    print("OK: target[d,y,x] and con head feature pair both refer to p -> p+(dy,dx).")


if __name__ == "__main__":
    main()

