"""Focused validation-only causal contrasts for the v6 directional dynamics."""

# ruff: noqa: E402

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Running this file directly puts diagnostics/, not the repository root, on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.road_comparison.road_metrics import (
    image_metrics,
    prepare_target,
    summarize,
)
from mass_protocol import blend_weight, sliding_positions
from networks.reliable_directional_repair import DIRECTIONS, dynamic_losses
from run_q3_r3_mass import PROTOCOL, full_dataset
from run_reliable_directional_mass import gap_lengths, load_model


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def work_mask(mask, size):
    return F.adaptive_max_pool2d(mask.float(), size) > 0


def ownership_work_mask(tile_size, work_size, local_bounds, device):
    y0, y1, x0, x1 = local_bounds
    sy, sx = tile_size[0] / work_size[0], tile_size[1] / work_size[1]
    yy = (torch.arange(work_size[0], device=device) + 0.5) * sy
    xx = (torch.arange(work_size[1], device=device) + 0.5) * sx
    return (
        (yy[:, None] >= y0)
        & (yy[:, None] < y1)
        & (xx[None, :] >= x0)
        & (xx[None, :] < x1)
    )[None, None]


def owner_bounds(positions, index, tile_size, image_size):
    start = positions[index]
    lower = (
        0
        if index == 0
        else int(
            round((positions[index - 1] + tile_size / 2 + start + tile_size / 2) / 2)
        )
    )
    upper = (
        image_size
        if index == len(positions) - 1
        else int(
            round((start + tile_size / 2 + positions[index + 1] + tile_size / 2) / 2)
        )
    )
    return max(lower - start, 0), min(upper - start, tile_size)


def tangent_and_valid(skeleton):
    work = work_mask(
        skeleton, (skeleton.shape[-2] // 4, skeleton.shape[-1] // 4)
    ).float()
    h, w = work.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=work.device, dtype=work.dtype),
        torch.arange(w, device=work.device, dtype=work.dtype),
        indexing="ij",
    )
    yy, xx = yy[None, None], xx[None, None]
    kernel = torch.ones((1, 1, 9, 9), device=work.device, dtype=work.dtype)
    count = F.conv2d(work, kernel, padding=4).clamp_min(1.0)
    sx, sy = (
        F.conv2d(work * xx, kernel, padding=4) / count,
        F.conv2d(work * yy, kernel, padding=4) / count,
    )
    sxx = F.conv2d(work * xx.square(), kernel, padding=4) / count - sx.square()
    syy = F.conv2d(work * yy.square(), kernel, padding=4) / count - sy.square()
    sxy = F.conv2d(work * xx * yy, kernel, padding=4) / count - sx * sy
    angle = 0.5 * torch.atan2(2.0 * sxy, sxx - syy)
    trace = (sxx + syy).clamp_min(0.0)
    separation = torch.sqrt((sxx - syy).square() + 4.0 * sxy.square())
    yy_near = F.conv2d(work, torch.ones((1, 1, 3, 3), device=work.device), padding=1)
    non_crossing = yy_near <= 3.01
    valid = (
        (work > 0) & (count >= 5.0) & non_crossing & (separation / (trace + 1e-6) > 0.5)
    )
    return angle, valid


def make_regions(result, mask, skeleton, threshold, ownership=None):
    size = result["u0"].shape[-2:]
    p0 = result["p0"]
    missing = (skeleton > 0.5) & (F.max_pool2d(p0, 5, 1, 2) < threshold)
    missing_work = work_mask(missing, size)
    p1_work = F.interpolate(
        result["p1"], size=size, mode="bilinear", align_corners=False
    )
    recovered = missing_work & (p1_work >= threshold)
    gt_work = work_mask(mask, size)
    road = (skeleton > 0.5) & (F.max_pool2d(p0, 5, 1, 2) >= threshold)
    reliable_road = work_mask(road, size)
    background_q = (mask < 0.5) & (
        F.interpolate(
            result["q"], size=mask.shape[-2:], mode="bilinear", align_corners=False
        )
        >= 0.5
    )
    hard_background = work_mask(background_q, size)
    new_fp = (result["p1"] >= threshold) & (p0 < threshold) & (mask < 0.5)
    isolated = new_fp & (F.max_pool2d((p0 >= threshold).float(), 7, 1, 3) == 0)
    bridge_batches = []
    for batch_index in range(mask.shape[0]):
        old = (p0[batch_index, 0].detach().cpu().numpy() >= threshold).astype(np.uint8)
        component_count, component_labels = cv2.connectedComponents(old, connectivity=8)
        new_pixels = new_fp[batch_index, 0].detach().cpu().numpy()
        touched = np.zeros_like(new_pixels, dtype=np.uint16)
        for component in range(1, component_count):
            near = (
                cv2.dilate(
                    (component_labels == component).astype(np.uint8),
                    np.ones((5, 5), np.uint8),
                )
                > 0
            )
            touched += (near & new_pixels).astype(np.uint8)
        bridge = new_pixels & (touched >= 2)
        bridge_batches.append(torch.from_numpy(bridge[None, None]).to(mask.device))
    false_bridge = torch.cat(bridge_batches, dim=0).bool()
    regions = {
        "recovered_gt": recovered,
        "unrecovered_gt": missing_work & ~recovered,
        "reliable_road": reliable_road,
        "hard_background": hard_background,
        "isolated_new_fp": work_mask(isolated, size),
        "false_bridge": work_mask(false_bridge, size),
        "gt_background": ~gt_work,
        "gt_road": gt_work,
        "missing_gt": missing_work,
    }
    if ownership is not None:
        regions = {name: region & ownership for name, region in regions.items()}
    return regions


def direction_stats(result, skeleton, regions, case):
    angle, valid = tangent_and_valid(skeleton)
    w = result["w"]
    direction_angles = torch.tensor(
        [np.arctan2(dy, dx) for dy, dx in DIRECTIONS], device=w.device, dtype=w.dtype
    ).view(1, 8, 1, 1)
    axis_score = torch.cos(2.0 * (direction_angles - angle))
    entropy = -(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(1, keepdim=True)
    alpha = result["alpha"].mean(1)
    rows = []
    for name in ("recovered_gt", "unrecovered_gt", "reliable_road"):
        region = regions[name] & valid
        if not region.any():
            rows.append({"case": case, "region": name, "valid_work_sites": 0})
            continue
        scores = axis_score.expand_as(w)
        alignment = (w * scores).sum(1, keepdim=True)
        directional_error = (
            0.5 * torch.acos(axis_score.clamp(-1.0, 1.0)) * (180.0 / np.pi)
        )
        expected_error = (w * directional_error).sum(1, keepdim=True)
        rows.append(
            {
                "case": case,
                "region": name,
                "valid_work_sites": int(region.sum()),
                "axis_consistency": float(alignment[region].mean()),
                "weighted_angle_error_deg": float(expected_error[region].mean()),
                "transport_weighted_axis_consistency": float(
                    (alignment * alpha)[region].sum()
                    / alpha[region].sum().clamp_min(1e-8)
                ),
                "angle_error_p50_deg": float(
                    torch.quantile(expected_error[region], 0.5)
                ),
                "angle_error_p90_deg": float(
                    torch.quantile(expected_error[region], 0.9)
                ),
                "w_entropy_mean": float((entropy / np.log(8.0))[region].mean()),
                "w_max_mean": float(w.amax(1, keepdim=True)[region].mean()),
                "w_max_p50": float(
                    torch.quantile(w.amax(1, keepdim=True)[region], 0.5)
                ),
                "alpha_mean": float(alpha[region].mean()),
            }
        )
    return rows


def forward_variant(
    model,
    image,
    baseline,
    mode,
    gt_work=None,
    capture=False,
    freeze_generated_sources=False,
):
    w = baseline["w"]
    gate = baseline["g_transport"]
    if mode == "uniform_w":
        w = torch.full_like(w, 1.0 / 8.0)
    elif mode == "rotated90_w":
        index = {offset: i for i, offset in enumerate(DIRECTIONS)}
        permutation = [index[(dx, -dy)] for dy, dx in DIRECTIONS]
        w = w[:, permutation]
    if mode == "oracle_g_background":
        gate = gate * gt_work.float()
    dynamics_output = model.dynamics(
        baseline["visual"],
        baseline["u0"],
        baseline["q"],
        baseline["r"],
        baseline["b"],
        baseline["a"],
        w,
        transport_gate=gate,
        capture_diagnostics=capture,
        freeze_generated_sources=freeze_generated_sources,
    )
    states, masses, availability = dynamics_output[:3]
    from networks.reliable_directional_repair import compose_surface_logits

    z1, _ = compose_surface_logits(baseline["z0"], baseline["u0"], states[-1])
    if not capture:
        return z1
    details = {key: value for key, value in dynamics_output[3].items()}
    details["source_mass"] = torch.stack(masses, dim=1)
    details["source_available"] = torch.stack(availability, dim=1)
    return z1, details


@torch.no_grad()
def tile_outputs(model, image, gt_tile, skeleton_tile, threshold):
    baseline = model(image, capture_diagnostics=True)
    size = baseline["u0"].shape[-2:]
    gt_work = work_mask(gt_tile, size)
    variants = {
        "baseline": baseline["z1"],
        "uniform_w": forward_variant(model, image, baseline, "uniform_w"),
        "rotated90_w": forward_variant(model, image, baseline, "rotated90_w"),
        "oracle_g_background": forward_variant(
            model, image, baseline, "oracle_g_background", gt_work
        ),
    }
    frozen_z1, frozen_dynamics = forward_variant(
        model,
        image,
        baseline,
        "original_w",
        capture=True,
        freeze_generated_sources=True,
    )
    frozen_result = dict(baseline)
    frozen_result.update(frozen_dynamics)
    frozen_result["z1"] = frozen_z1
    frozen_result["p1"] = frozen_z1.sigmoid()
    variants["frozen_r_sources"] = frozen_z1
    oracle_r = baseline["r"] * gt_work.float()
    oracle_r_result = model(image, capture_diagnostics=True, reliable_override=oracle_r)
    variants["oracle_clear_r_background"] = oracle_r_result["z1"]
    _, oracle_g_dynamics = forward_variant(
        model, image, baseline, "oracle_g_background", gt_work, capture=True
    )
    diagnostic_modes = {
        "baseline": baseline,
        "frozen_r_sources": frozen_result,
        "oracle_clear_r_background": oracle_r_result,
        "oracle_g_background": oracle_g_dynamics,
    }
    return baseline, variants, diagnostic_modes


def cpu_snapshot(result, keys):
    return {
        key: result[key].detach().cpu()
        for key in keys
        if key in result and torch.is_tensor(result[key])
    }


def stitch_image(model, image, gt, skeleton, device, threshold, case_name):
    tile_size, stride = 512, 256
    weight = blend_weight(tile_size, device)
    positions = sliding_positions(image.shape[-1], tile_size, stride)
    names = (
        "baseline",
        "uniform_w",
        "rotated90_w",
        "frozen_r_sources",
        "oracle_clear_r_background",
        "oracle_g_background",
    )
    canvases = {
        name: torch.zeros((1, 1, *image.shape[-2:]), device=device)
        for name in ("p0", *names)
    }
    denom = torch.zeros_like(canvases["p0"])
    diagnostic_names = ("q", "r", "c_bridge", "c_support", "g_transport")
    diagnostic_canvases = {
        name: torch.zeros_like(canvases["p0"]) for name in diagnostic_names
    }
    chosen = None
    tile_records = []
    baseline_keys = (
        "p0",
        "p1",
        "u0",
        "u1",
        "u2",
        "u3",
        "w",
        "r",
        "q",
        "g_transport",
        "c_bridge",
        "c_support",
        "a",
        "alpha",
        "beta",
        "gamma",
        "stay",
        "source_available",
        "source_mass",
        "transport_delta",
        "proposal_delta",
        "anchor_delta",
        "proposal_raw",
        "delta_z_plus",
        "delta_z_minus",
    )
    mode_keys = ("source_available", "alpha", "transport_delta")
    for row_index, top in enumerate(positions):
        for col_index, left in enumerate(positions):
            tile = image[:, :, top : top + tile_size, left : left + tile_size]
            gt_tile = gt[:, :, top : top + tile_size, left : left + tile_size]
            sk_tile = skeleton[:, :, top : top + tile_size, left : left + tile_size]
            base, variants, diagnostic_modes = tile_outputs(
                model, tile, gt_tile, sk_tile, threshold
            )
            region = (..., slice(top, top + tile_size), slice(left, left + tile_size))
            canvases["p0"][region] += base["z0"].float() * weight
            for name, logits in variants.items():
                canvases[name][region] += logits.float() * weight
            for name in diagnostic_names:
                value = base[name]
                if value.shape[-2:] != (tile_size, tile_size):
                    value = F.interpolate(
                        value.float(),
                        size=(tile_size, tile_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                diagnostic_canvases[name][region] += value.float() * weight
            denom[region] += weight
            owner_y0, owner_y1 = owner_bounds(
                positions, row_index, tile_size, image.shape[-2]
            )
            owner_x0, owner_x1 = owner_bounds(
                positions, col_index, tile_size, image.shape[-1]
            )
            local_owner = (owner_y0, owner_y1, owner_x0, owner_x1)
            tile_records.append(
                {
                    "top": top,
                    "left": left,
                    "owner": local_owner,
                    "base": cpu_snapshot(base, baseline_keys),
                    "modes": {
                        mode: cpu_snapshot(value, mode_keys)
                        for mode, value in diagnostic_modes.items()
                    },
                }
            )
            gap_count = int(
                (
                    (sk_tile > 0.5) & (F.max_pool2d(base["p0"], 5, 1, 2) < threshold)
                ).sum()
            )
            if chosen is None or gap_count > chosen[0]:
                chosen = (
                    gap_count,
                    top,
                    left,
                    local_owner,
                )
    if (denom <= 0).any():
        raise RuntimeError("Stitching left uncovered pixels")
    arrays = {
        name: (value / denom).float().cpu().numpy()[0, 0]
        for name, value in canvases.items()
    }
    arrays.update(
        {
            name: (value / denom).float().cpu().numpy()[0, 0]
            for name, value in diagnostic_canvases.items()
        }
    )
    if chosen is None:
        raise RuntimeError("No representative validation tile was selected")
    (gap_count, top, left, local_owner) = chosen
    tile_id = f"{top}_{left}"
    all_directions, all_sources, all_saturation = [], [], []
    all_source_modes, all_gate_rows = [], []
    p0_full = torch.from_numpy(
        1.0 / (1.0 + np.exp(-np.clip(arrays["p0"], -80.0, 80.0)))
    ).to(device)[None, None]
    p1_full = torch.from_numpy(
        1.0 / (1.0 + np.exp(-np.clip(arrays["baseline"], -80.0, 80.0)))
    ).to(device)[None, None]
    for record in tile_records:
        top_i, left_i = record["top"], record["left"]
        ownership = ownership_work_mask(
            (tile_size, tile_size),
            record["base"]["u0"].shape[-2:],
            record["owner"],
            device,
        )
        base = {key: value.to(device) for key, value in record["base"].items()}
        ref = (..., slice(top_i, top_i + tile_size), slice(left_i, left_i + tile_size))
        base["p0"] = p0_full[ref]
        base["p1"] = p1_full[ref]
        gt_tile = gt[:, :, top_i : top_i + tile_size, left_i : left_i + tile_size]
        sk_tile = skeleton[:, :, top_i : top_i + tile_size, left_i : left_i + tile_size]
        regions = make_regions(base, gt_tile, sk_tile, threshold, ownership)
        record_tile_id = f"{top_i}_{left_i}"
        owner_y0, owner_y1, owner_x0, owner_x1 = record["owner"]
        ownership_full = torch.zeros(
            (tile_size, tile_size), dtype=torch.bool, device=device
        )
        ownership_full[owner_y0:owner_y1, owner_x0:owner_x1] = True
        all_directions.extend(direction_stats(base, sk_tile, regions, record_tile_id))
        all_sources.extend(
            trace_sources(
                base,
                gt_tile,
                sk_tile,
                record_tile_id,
                top_i,
                left_i,
                threshold,
                ownership_full,
            )
        )
        all_saturation.extend(
            saturation_stats(
                base,
                gt_tile,
                sk_tile,
                record_tile_id,
                threshold=threshold,
                ownership=ownership,
            )
        )
        mode_results = {
            name: {key: value.to(device) for key, value in values.items()}
            for name, values in record["modes"].items()
        }
        all_source_modes.extend(
            source_dynamics_rows(mode_results, regions, record_tile_id)
        )
        all_gate_rows.extend(gate_region_stats(base, regions, record_tile_id))

    selected_image = image[:, :, top : top + tile_size, left : left + tile_size]
    selected_gt = gt[:, :, top : top + tile_size, left : left + tile_size]
    selected_sk = skeleton[:, :, top : top + tile_size, left : left + tile_size]
    with torch.enable_grad():
        grad_result = model(selected_image, capture_diagnostics=True)
        total_loss, _ = dynamic_losses(
            grad_result, selected_gt, selected_sk, **model.audit_loss_kwargs
        )
        raw_gradient = torch.autograd.grad(
            total_loss, grad_result["proposal_raw"], allow_unused=True
        )[0]
    grad_result["p0"] = p0_full[..., top : top + tile_size, left : left + tile_size]
    grad_result["p1"] = p1_full[..., top : top + tile_size, left : left + tile_size]
    selected_ownership = ownership_work_mask(
        (tile_size, tile_size), grad_result["u0"].shape[-2:], local_owner, device
    )
    gradient_rows = saturation_stats(
        grad_result,
        selected_gt,
        selected_sk,
        tile_id,
        raw_gradient,
        threshold,
        selected_ownership,
    )
    saturation_by_region = {
        row["region"]: row for row in all_saturation if row["case"] == tile_id
    }
    for row in gradient_rows:
        target = saturation_by_region.get(row["region"])
        if target is not None:
            for key, value in row.items():
                if key.endswith("gradient_mean_abs") or key.endswith(
                    "gradient_zero_fraction"
                ):
                    target[key] = value
            target["gradient_audit_scope"] = "one owned representative tile per image"
    pair_rows, matched_pair_rows, pair_summary = connection_pair_rows(
        arrays,
        {"case_name": case_name, "mask": gt[0], "skeleton": skeleton[0]},
        threshold,
    )
    return (
        arrays,
        all_directions,
        pair_rows,
        matched_pair_rows,
        pair_summary,
        all_sources,
        all_saturation,
        all_source_modes,
        all_gate_rows,
    )


def _endpoints(binary):
    kernel = np.ones((3, 3), dtype=np.uint8)
    road = binary.astype(np.uint8)
    neighbors = np.rint(
        cv2.filter2D(road.astype(np.float32), cv2.CV_32F, kernel)
    ).astype(np.int16) - road.astype(np.int16)
    return np.argwhere(binary & (neighbors == 1))


def _nearest(points, query, max_distance):
    if len(points) == 0:
        return None, np.inf
    delta = points.astype(np.float32) - np.asarray(query, dtype=np.float32)[None]
    distances = np.sqrt((delta * delta).sum(1))
    index = int(distances.argmin())
    return (tuple(map(int, points[index])), float(distances[index]))


def _line_points(a, b):
    y0, x0 = min(a[0], b[0]), min(a[1], b[1])
    y1, x1 = max(a[0], b[0]), max(a[1], b[1])
    line = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.line(line, (a[1] - x0, a[0] - y0), (b[1] - x0, b[0] - y0), 1, 1)
    yy, xx = np.nonzero(line)
    return np.stack((yy + y0, xx + x0), axis=1)


def _pair_row(case, candidate_id, kind, label, endpoints, route, fields):
    h, w = fields["p0"].shape
    route = np.asarray(route, dtype=np.int32)
    valid = (
        (route[:, 0] >= 0) & (route[:, 0] < h) & (route[:, 1] >= 0) & (route[:, 1] < w)
    )
    route = route[valid]
    if len(route) == 0:
        return None
    end_a, end_b = endpoints
    values = {}
    for name in ("p0", "q3", "r"):
        image = fields[name]
        values[f"{name}_endpoint_mean"] = float(
            (image[tuple(end_a)] + image[tuple(end_b)]) * 0.5
        )
        values[f"{name}_path_mean"] = float(image[route[:, 0], route[:, 1]].mean())
    for name in ("c_bridge", "g_transport", "c_support"):
        sampled = fields[name][route[:, 0], route[:, 1]]
        values[f"{name}_path_mean"] = float(sampled.mean())
        values[f"{name}_path_p10"] = float(np.quantile(sampled, 0.1))
        values[f"{name}_path_min"] = float(sampled.min())
    distance = float(np.linalg.norm(np.asarray(end_a) - np.asarray(end_b)))
    p_bin = int(values["p0_endpoint_mean"] / 0.2)
    q_bin = int(values["q3_endpoint_mean"] / 0.2)
    r_bin = int(values["r_endpoint_mean"] / 0.2)
    distance_bin = min(int(distance // 16), 8)
    return {
        "case": case,
        "candidate_id": candidate_id,
        "candidate_kind": kind,
        "label_true_missing_connection": int(label),
        "endpoint_a_y": int(end_a[0]),
        "endpoint_a_x": int(end_a[1]),
        "endpoint_b_y": int(end_b[0]),
        "endpoint_b_x": int(end_b[1]),
        "endpoint_distance_px": distance,
        "corridor_sample_pixels": int(len(route)),
        "stratum": f"{distance_bin}:{p_bin}:{q_bin}:{r_bin}",
        "matched": False,
        **values,
    }


def connection_pair_rows(arrays, sample, threshold):
    """Build GT-path positives and nearby disconnected-GT endpoint negatives."""
    from scipy.spatial import cKDTree

    case = sample["case_name"]
    skeleton = to_numpy(sample["skeleton"][0]) > 0.5
    p0 = 1.0 / (1.0 + np.exp(-np.clip(arrays["p0"], -80, 80)))
    support = (
        cv2.dilate((p0 >= threshold).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    )
    missing = skeleton & ~support
    matched_skeleton = skeleton & support
    gt_count, gt_labels = cv2.connectedComponents(
        skeleton.astype(np.uint8), connectivity=8
    )
    missing_count, missing_labels, missing_stats, _ = cv2.connectedComponentsWithStats(
        missing.astype(np.uint8), connectivity=8
    )
    fields = {
        "p0": p0,
        "q3": arrays["q"],
        "r": arrays["r"],
        "c_bridge": arrays["c_bridge"],
        "g_transport": arrays["g_transport"],
        "c_support": arrays["c_support"],
    }
    rows, reference_gaps, rejected_gaps = [], 0, 0
    positive_id = 0
    for index in range(1, missing_count):
        if missing_stats[index, cv2.CC_STAT_AREA] < 3:
            continue
        reference_gaps += 1
        component = missing_labels == index
        endpoints = _endpoints(component)
        if len(endpoints) < 2 or len(endpoints) > 8:
            rejected_gaps += 1
            continue
        # Select the two most separated ends; branches/ambiguous gaps are excluded.
        endpoint_pairs = (
            (float(np.sum((a - b) ** 2)), a, b)
            for i, a in enumerate(endpoints)
            for b in endpoints[i + 1 :]
        )
        _, gap_a, gap_b = max(endpoint_pairs, key=lambda item: item[0])
        label_a = int(gt_labels[tuple(gap_a)])
        label_b = int(gt_labels[tuple(gap_b)])
        if label_a == 0 or label_a != label_b:
            rejected_gaps += 1
            continue
        anchors = np.argwhere(matched_skeleton & (gt_labels == label_a))
        anchor_a, dist_a = _nearest(anchors, gap_a, 20)
        anchor_b, dist_b = _nearest(anchors, gap_b, 20)
        if (
            anchor_a is None
            or anchor_b is None
            or dist_a > 20
            or dist_b > 20
            or anchor_a == anchor_b
        ):
            rejected_gaps += 1
            continue
        route = np.argwhere(component)
        candidate_id = positive_id + 1
        row = _pair_row(
            case,
            f"pos_{candidate_id:05d}",
            "gt_missing_path",
            1,
            (anchor_a, anchor_b),
            route,
            fields,
        )
        if row is not None:
            positive_id = candidate_id
            row["reference_gap_pixels"] = int(len(route))
            row["flank_match_distance_a_px"] = dist_a
            row["flank_match_distance_b_px"] = dist_b
            rows.append(row)
        else:
            rejected_gaps += 1

    # Nearby endpoints from distinct GT connected components are hard negatives.
    endpoint_records = []
    for component_id in range(1, gt_count):
        points = _endpoints(gt_labels == component_id)
        if len(points) > 128:
            points = points[np.linspace(0, len(points) - 1, 128).astype(int)]
        endpoint_records.extend(
            (component_id, tuple(map(int, point))) for point in points
        )
    negative_candidates = []
    if len(endpoint_records) > 1:
        coords = np.asarray([item[1] for item in endpoint_records], dtype=np.float32)
        tree = cKDTree(coords)
        pair_set = set()
        _, neighbors = tree.query(
            coords, k=min(9, len(endpoint_records)), distance_upper_bound=128.0
        )
        for idx, (component_a, point_a) in enumerate(endpoint_records):
            for j in np.atleast_1d(neighbors[idx]):
                if not np.isfinite(j) or int(j) >= len(endpoint_records):
                    continue
                j = int(j)
                component_b, point_b = endpoint_records[j]
                if component_a == component_b or j == idx:
                    continue
                pair = tuple(sorted((idx, j)))
                if pair in pair_set:
                    continue
                distance = float(np.linalg.norm(coords[idx] - coords[j]))
                if distance < 6 or distance > 128:
                    continue
                pair_set.add(pair)
                negative_candidates.append((distance, idx, j, component_a, component_b))
        negative_candidates.sort()
    seen_component_pairs, negative_id = {}, 0
    for distance, idx, j, component_a, component_b in negative_candidates:
        component_pair = tuple(sorted((component_a, component_b)))
        if seen_component_pairs.get(component_pair, 0) >= 3 or negative_id >= 2000:
            continue
        point_a, point_b = endpoint_records[idx][1], endpoint_records[j][1]
        route = _line_points(point_a, point_b)
        negative_id += 1
        row = _pair_row(
            case,
            f"neg_{negative_id:05d}",
            "nearby_disconnected_gt_components",
            0,
            (point_a, point_b),
            route,
            fields,
        )
        if row is not None:
            row["gt_component_a"] = component_a
            row["gt_component_b"] = component_b
            rows.append(row)
            seen_component_pairs[component_pair] = (
                seen_component_pairs.get(component_pair, 0) + 1
            )

    groups = {}
    for row in rows:
        groups.setdefault(row["stratum"], {0: [], 1: []})[
            row["label_true_missing_connection"]
        ].append(row)
    rng = np.random.default_rng(921)
    matched_rows = []
    for stratum, group in groups.items():
        count = min(len(group[0]), len(group[1]), 100)
        if count == 0:
            continue
        for label in (0, 1):
            selected = rng.choice(len(group[label]), count, replace=False)
            for selected_index in selected:
                row = dict(group[label][int(selected_index)])
                row["matched"] = True
                row["matched_stratum"] = stratum
                matched_rows.append(row)
    summary = {
        "case": case,
        "reference_missing_segments_ge3px": reference_gaps,
        "positive_pairs_generated": positive_id,
        "positive_pairs_rejected": rejected_gaps,
        "positive_pair_generation_recall": positive_id / max(reference_gaps, 1),
        "negative_pairs_generated": negative_id,
        "matched_positive_pairs": sum(
            row["label_true_missing_connection"] == 1 for row in matched_rows
        ),
        "matched_negative_pairs": sum(
            row["label_true_missing_connection"] == 0 for row in matched_rows
        ),
    }
    return rows, matched_rows, summary


def trace_sources(
    result, mask, skeleton, case, tile_top, tile_left, threshold, ownership_full=None
):
    size = result["u0"].shape[-2:]
    gt_work = work_mask(mask, size)[0, 0]
    p0 = result["p0"][0, 0]
    p1 = result["p1"][0, 0]
    new_fp = (p1 >= threshold) & (p0 < threshold) & (mask[0, 0] < 0.5)
    if ownership_full is not None:
        new_fp &= ownership_full
    count, labels = cv2.connectedComponents(
        (p0.detach().cpu().numpy() >= threshold).astype(np.uint8), connectivity=8
    )
    candidate = new_fp.detach().cpu().numpy().astype(np.uint8)
    neighbor_count = np.zeros_like(candidate, dtype=np.uint8)
    for component in range(1, count):
        near = (
            cv2.dilate(
                (labels == component).astype(np.uint8), np.ones((5, 5), np.uint8)
            )
            > 0
        )
        neighbor_count += (near & candidate.astype(bool)).astype(np.uint8)
    bridge = torch.from_numpy((candidate.astype(bool) & (neighbor_count >= 2))).to(
        p0.device
    )
    isolated = torch.from_numpy((candidate.astype(bool) & (neighbor_count == 0))).to(
        p0.device
    )
    bridge_work = work_mask(bridge[None, None].float(), size)[0, 0]
    isolated_work = work_mask(isolated[None, None].float(), size)[0, 0]
    target_mask = bridge_work | isolated_work
    target_positions = torch.nonzero(target_mask, as_tuple=False)
    if not target_positions.numel():
        return []
    transport_rank = result["transport_delta"][0].sum((0, 1)).clamp_min(0.0)
    positions = target_positions.tolist()
    positions.sort(
        key=lambda point: float(transport_rank[point[0], point[1]]), reverse=True
    )
    records = []
    w, r = result["w"][0], result["r"][0]
    scale_y = p0.shape[-2] / size[0]
    scale_x = p0.shape[-1] / size[1]
    for y, x in positions[:100]:
        for step in range(3):
            state = result[f"u{step}"][0, 0]
            gained = (
                (state - result["u0"][0, 0]).clamp_min(0.0)
                / (1.0 - result["u0"][0, 0]).clamp_min(1e-5)
            ).clamp_max(1.0)
            support = r[0] + (1.0 - r[0]) * gained
            donors = []
            mass = result["source_mass"][0, step, 0, y, x].clamp_min(1e-8)
            alpha = result["alpha"][0, step, 0, y, x]
            for direction, (dy, dx) in enumerate(DIRECTIONS):
                sy, sx = y + dy, x + dx
                if not (0 <= sy < size[0] and 0 <= sx < size[1]):
                    continue
                source_weight = float(w[direction, y, x] * support[sy, sx])
                contribution = float(
                    alpha * source_weight / mass * (state[sy, sx] - state[y, x])
                )
                donors.append(
                    (
                        abs(contribution),
                        source_weight,
                        direction,
                        sy,
                        sx,
                        float(support[sy, sx]),
                        contribution,
                    )
                )
            if not donors:
                continue
            _, source_weight, direction, sy, sx, support_value, contribution = max(
                donors
            )
            dy, dx = DIRECTIONS[direction]
            records.append(
                {
                    "case": case,
                    "target_type": "false_bridge"
                    if bool(bridge_work[y, x])
                    else "isolated_new_fp",
                    "target_y_work": y,
                    "target_x_work": x,
                    "target_y_tile": int((y + 0.5) * scale_y),
                    "target_x_tile": int((x + 0.5) * scale_x),
                    "target_y_image": tile_top + int((y + 0.5) * scale_y),
                    "target_x_image": tile_left + int((x + 0.5) * scale_x),
                    "step": step + 1,
                    "source_direction": direction,
                    "source_offset_dy": dy,
                    "source_offset_dx": dx,
                    "source_y_work": sy,
                    "source_x_work": sx,
                    "source_y_image": tile_top + int((sy + 0.5) * scale_y),
                    "source_x_image": tile_left + int((sx + 0.5) * scale_x),
                    "source_gt_road": int(bool(gt_work[sy, sx])),
                    "source_r": float(r[0, sy, sx]),
                    "source_state": float(state[sy, sx]),
                    "source_support": support_value,
                    "source_weight": source_weight,
                    "target_alpha": float(alpha),
                    "target_transport_delta": float(
                        result["transport_delta"][0, step, 0, y, x]
                    ),
                    "donor_transport_contribution": contribution,
                    "target_g_transport": float(result["g_transport"][0, 0, y, x]),
                }
            )
    return records


def source_dynamics_rows(mode_results, regions, case):
    rows = []
    for mode, result in mode_results.items():
        for region_name in (
            "recovered_gt",
            "unrecovered_gt",
            "isolated_new_fp",
            "false_bridge",
        ):
            region = regions[region_name]
            row = {
                "case": case,
                "mode": mode,
                "region": region_name,
                "work_pixels": int(region.sum()),
            }
            for step in range(3):
                row[f"v{step + 1}_mean"] = (
                    float(result["source_available"][0, step, 0][region[0, 0]].mean())
                    if region.any()
                    else None
                )
                row[f"alpha{step + 1}_mean"] = (
                    float(result["alpha"][0, step, 0][region[0, 0]].mean())
                    if region.any()
                    else None
                )
                row[f"transport_delta{step + 1}_signed"] = (
                    float(result["transport_delta"][0, step, 0][region[0, 0]].mean())
                    if region.any()
                    else None
                )
                row[f"transport_delta{step + 1}_abs"] = (
                    float(
                        result["transport_delta"][0, step, 0][region[0, 0]].abs().mean()
                    )
                    if region.any()
                    else None
                )
            rows.append(row)
    return rows


def saturation_stats(
    result, mask, skeleton, case, raw_gradient=None, threshold=0.45, ownership=None
):
    raw = result["proposal_raw"]
    plus, minus = result["delta_z_plus"], result["delta_z_minus"]
    regions = make_regions(result, mask, skeleton, threshold, ownership)
    rows = []
    for name, region in regions.items():
        if name not in (
            "recovered_gt",
            "unrecovered_gt",
            "isolated_new_fp",
            "false_bridge",
            "hard_background",
        ):
            continue
        values_plus, values_minus = plus[region], minus[region]
        raw_plus, raw_minus = raw[:, :1][region], raw[:, 1:2][region]
        if not values_plus.numel():
            continue
        deriv_plus = 8.0 * (values_plus / 8.0) * (1.0 - values_plus / 8.0)
        deriv_minus = 4.0 * (values_minus / 4.0) * (1.0 - values_minus / 4.0)
        grad_plus = raw_gradient[:, :1][region] if raw_gradient is not None else None
        grad_minus = raw_gradient[:, 1:2][region] if raw_gradient is not None else None
        rows.append(
            {
                "case": case,
                "region": name,
                "work_sites": int(values_plus.numel()),
                "raw_plus_p50": float(torch.quantile(raw_plus, 0.5)),
                "raw_plus_p90": float(torch.quantile(raw_plus, 0.9)),
                "delta_plus_p50": float(torch.quantile(values_plus, 0.5)),
                "delta_plus_p90": float(torch.quantile(values_plus, 0.9)),
                "delta_plus_ge_7_9": float((values_plus >= 7.9).float().mean()),
                "delta_plus_ge_7_99": float((values_plus >= 7.99).float().mean()),
                "delta_plus_ge_7_999": float((values_plus >= 7.999).float().mean()),
                "delta_plus_at_8": float((values_plus >= 8.0).float().mean()),
                "plus_derivative_mean": float(deriv_plus.mean()),
                "plus_derivative_zero_fraction": float(
                    (deriv_plus < 1e-5).float().mean()
                ),
                "plus_loss_gradient_mean_abs": (
                    float(grad_plus.abs().mean()) if grad_plus is not None else None
                ),
                "plus_loss_gradient_zero_fraction": (
                    float((grad_plus.abs() < 1e-12).float().mean())
                    if grad_plus is not None
                    else None
                ),
                "raw_minus_p50": float(torch.quantile(raw_minus, 0.5)),
                "raw_minus_p90": float(torch.quantile(raw_minus, 0.9)),
                "delta_minus_mean": float(values_minus.mean()),
                "delta_minus_p90": float(torch.quantile(values_minus, 0.9)),
                "minus_derivative_mean": float(deriv_minus.mean()),
                "minus_loss_gradient_mean_abs": (
                    float(grad_minus.abs().mean()) if grad_minus is not None else None
                ),
                "minus_loss_gradient_zero_fraction": (
                    float((grad_minus.abs() < 1e-12).float().mean())
                    if grad_minus is not None
                    else None
                ),
                "minus_near_zero_fraction": float((values_minus < 0.01).float().mean()),
            }
        )
    return rows


def gate_region_stats(result, regions, case):
    rows = []
    for name in ("recovered_gt", "unrecovered_gt", "isolated_new_fp", "false_bridge"):
        region = regions[name]
        row = {"case": case, "region": name, "work_sites": int(region.sum())}
        for field in ("g_transport", "c_bridge", "c_support", "r", "a"):
            values = result[field][region]
            for quantile, suffix in ((0.1, "p10"), (0.5, "p50"), (0.9, "p90")):
                row[f"{field}_{suffix}"] = (
                    float(torch.quantile(values.float(), quantile))
                    if values.numel()
                    else None
                )
        rows.append(row)
    return rows


def image_level_metrics(arrays, sample, threshold):
    gt = to_numpy(sample["mask"][0]) > 0.5
    prepared = prepare_target(gt, PROTOCOL)
    rows = []
    p0_prob = 1.0 / (1.0 + np.exp(-np.clip(arrays["p0"], -80.0, 80.0)))
    p0_binary = p0_prob >= threshold
    for name in (
        "p0",
        "baseline",
        "uniform_w",
        "rotated90_w",
        "frozen_r_sources",
        "oracle_clear_r_background",
        "oracle_g_background",
    ):
        prob = 1.0 / (1.0 + np.exp(-np.clip(arrays[name], -80.0, 80.0)))
        pred = prob >= threshold
        metrics = image_metrics(pred, prepared, PROTOCOL)
        bins, bridge = gap_lengths(p0_binary, pred, prepared["skeleton"], gt)
        eligible = sum(item["eligible"] for item in bins.values())
        closed = sum(item["closed"] for item in bins.values())
        rows.append(
            {
                "mode": name,
                "case_id": sample["case_name"],
                **metrics,
                "eligible_gaps": eligible,
                "closed_gaps": closed,
                "gap_closure_rate": closed / max(eligible, 1),
                "background_bridge_proxy": bridge,
                "new_tp": int((pred & ~p0_binary & gt).sum()),
                "new_fp": int((pred & ~p0_binary & ~gt).sum()),
                "isolated_new_fp_pixels": int(
                    (
                        pred
                        & ~p0_binary
                        & ~gt
                        & ~(
                            cv2.dilate(
                                p0_binary.astype(np.uint8), np.ones((7, 7), np.uint8)
                            )
                            > 0
                        )
                    ).sum()
                ),
            }
        )
    return rows


def save_case_image(path, sample, arrays):
    image = to_numpy(sample["image"]).transpose(1, 2, 0)
    image = np.uint8(
        np.clip((image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255, 0, 255)
    )
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gt = to_numpy(sample["mask"][0]) > 0.5
    panels = [("image", image), ("GT", gt.astype(np.float32))]
    for name in (
        "p0",
        "baseline",
        "uniform_w",
        "rotated90_w",
        "frozen_r_sources",
        "oracle_clear_r_background",
        "oracle_g_background",
    ):
        value = arrays[name]
        if name != "p0":
            value = 1.0 / (1.0 + np.exp(-np.clip(value, -80.0, 80.0)))
        else:
            value = 1.0 / (1.0 + np.exp(-np.clip(value, -80.0, 80.0)))
        panels.append((name, value))
    bitmaps = []
    for title, value in panels:
        if value.ndim == 2:
            value = cv2.applyColorMap(
                np.uint8(np.clip(value, 0, 1) * 255), cv2.COLORMAP_TURBO
            )
        value = cv2.resize(value, (375, 375), interpolation=cv2.INTER_AREA)
        cv2.putText(
            value, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )
        bitmaps.append(value)
    canvas = np.concatenate(
        (
            np.concatenate(bitmaps[:3], axis=1),
            np.concatenate(bitmaps[3:6], axis=1),
            np.concatenate(bitmaps[6:], axis=1),
        ),
        axis=0,
    )
    cv2.imwrite(str(path), canvas)


def matched_ranking(rows, score_name):
    positive = np.asarray(
        [
            row[score_name]
            for row in rows
            if row["label_true_missing_connection"] == 1 and row["matched"]
        ]
    )
    negative = np.asarray(
        [
            row[score_name]
            for row in rows
            if row["label_true_missing_connection"] == 0 and row["matched"]
        ]
    )
    if not positive.size or not negative.size or positive.size != negative.size:
        return {
            "roc_auc": None,
            "average_precision": None,
            "positive_count": int(positive.size),
            "negative_count": int(negative.size),
        }
    values = np.concatenate((positive, negative))
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    auc = (ranks[: len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (
        len(positive) ** 2
    )
    descending = np.argsort(-values, kind="mergesort")
    sorted_values, sorted_labels = values[descending], labels[descending]
    true_positive = false_positive = 0
    previous_recall = 0.0
    ap = 0.0
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        group = sorted_labels[start:end]
        true_positive += int(group.sum())
        false_positive += int(len(group) - group.sum())
        recall = true_positive / len(positive)
        precision = true_positive / (true_positive + false_positive)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    ap = float(ap)
    return {
        "roc_auc": float(auc),
        "average_precision": ap,
        "positive_count": len(positive),
        "negative_count": len(negative),
    }


def run(args):
    if not args.root_path.is_dir():
        raise FileNotFoundError(
            f"Validation dataset directory not found: {args.root_path}"
        )
    if not args.dynamic_checkpoint.is_file():
        raise FileNotFoundError(
            "Dynamic checkpoint must be an existing file; got "
            f"{args.dynamic_checkpoint}. Check that V6_CKPT is set to the full best.pth path."
        )
    payload = torch.load(
        args.dynamic_checkpoint, map_location="cpu", weights_only=False
    )
    args.base_checkpoint = args.base_checkpoint or Path(payload["base_checkpoint"])
    args.q3_checkpoint = args.q3_checkpoint or Path(payload["q3_checkpoint"])
    for name, checkpoint in (
        ("base", args.base_checkpoint),
        ("Q3", args.q3_checkpoint),
    ):
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"{name} checkpoint from dynamic checkpoint metadata is missing: {checkpoint}"
            )
    config = SimpleNamespace(
        mode="audit",
        base_checkpoint=args.base_checkpoint.resolve(),
        q3_checkpoint=args.q3_checkpoint.resolve(),
        dynamic_checkpoint=args.dynamic_checkpoint.resolve(),
        device=torch.device(args.device),
        width=payload["width"],
        accept_floor_initial=payload["accept_floor_initial"],
        accept_floor_anneal_epochs=payload["accept_floor_anneal_epochs"],
        missing_threshold=payload["missing_threshold"],
        lambda_bg=payload["lambda_bg"],
        lambda_cont=payload["lambda_cont"],
        lambda_missing=payload.get("lambda_missing", 2.0),
    )
    model = load_model(config).eval()
    model.audit_loss_kwargs = {
        "lambda_bg": payload["lambda_bg"],
        "lambda_missing": payload.get("lambda_missing", 2.0),
        "lambda_cont": payload["lambda_cont"],
        "missing_threshold": payload["missing_threshold"],
    }
    dataset = full_dataset(args.root_path, "val")
    if len(dataset) < args.images:
        raise ValueError(
            f"Expected at least {args.images} val images, found {len(dataset)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics, all_directions, all_pairs, all_matched_pairs = [], [], [], []
    all_pair_summaries, all_sources, all_saturation = [], [], []
    all_source_modes, all_gate_rows = [], []
    for index in range(args.images):
        sample = dataset[index]
        image = sample["image"][None].to(args.device)
        gt = sample["mask"][None].to(args.device)
        skeleton = sample["skeleton"][None].to(args.device)
        (
            arrays,
            direction_rows,
            pair_rows,
            matched_pair_rows,
            pair_summary,
            source_rows,
            saturation_rows,
            source_mode_rows,
            gate_rows,
        ) = stitch_image(
            model, image, gt, skeleton, args.device, args.threshold, sample["case_name"]
        )
        for row in image_level_metrics(arrays, sample, args.threshold):
            row["case"] = sample["case_name"]
            all_metrics.append(row)
        all_directions.extend(
            {"image_case": sample["case_name"], **row} for row in direction_rows
        )
        all_pairs.extend(pair_rows)
        all_matched_pairs.extend(matched_pair_rows)
        all_pair_summaries.append(pair_summary)
        all_sources.extend(
            {"image_case": sample["case_name"], **row} for row in source_rows
        )
        all_saturation.extend(
            {"image_case": sample["case_name"], **row} for row in saturation_rows
        )
        all_source_modes.extend(
            {"image_case": sample["case_name"], **row} for row in source_mode_rows
        )
        all_gate_rows.extend(
            {"image_case": sample["case_name"], **row} for row in gate_rows
        )
        save_case_image(args.output_dir / f"case_{index:02d}.png", sample, arrays)
        print(
            f"[V6 CAUSAL] {index + 1}/{args.images} {sample['case_name']}", flush=True
        )
    write_csv(args.output_dir / "counterfactual_metrics.csv", all_metrics)
    write_csv(args.output_dir / "direction_alignment.csv", all_directions)
    write_csv(args.output_dir / "connection_pair_candidates.csv", all_pairs)
    write_csv(args.output_dir / "matched_connection_pairs.csv", all_matched_pairs)
    write_csv(args.output_dir / "connection_pair_generation.csv", all_pair_summaries)
    write_csv(args.output_dir / "source_trace.csv", all_sources)
    write_csv(args.output_dir / "delta_saturation.csv", all_saturation)
    write_csv(args.output_dir / "source_dynamics.csv", all_source_modes)
    write_csv(args.output_dir / "gate_regions.csv", all_gate_rows)
    matched = {
        name: matched_ranking(all_matched_pairs, name)
        for name in (
            "c_bridge_path_mean",
            "c_bridge_path_p10",
            "g_transport_path_mean",
            "g_transport_path_p10",
            "c_support_path_mean",
        )
    }
    mode_summary = {}
    for name in (
        "p0",
        "baseline",
        "uniform_w",
        "rotated90_w",
        "frozen_r_sources",
        "oracle_clear_r_background",
        "oracle_g_background",
    ):
        selected = [row for row in all_metrics if row["mode"] == name]
        metric_fields = (
            "tp",
            "fp",
            "fn",
            "topo_precision",
            "topo_recall",
            "cldice",
            "pred_components",
            "gt_components",
            "frag_idx",
            "fragment_density_per_1000_px",
            "short_pred_components",
            "short_gt_components",
            "extra_components",
            "largest_component_ratio",
            "gt_largest_component_ratio",
            "break_pixels",
            "break_rate",
            "gap_components",
            "max_gap_pixels",
            "apls_gt_to_pred_approx",
            "apls_pred_to_gt_approx",
            "apls_bidirectional_approx",
            "apls_gt_valid_pairs",
            "apls_pred_valid_pairs",
        )
        metrics = summarize(
            [{key: row[key] for key in metric_fields} for row in selected]
        )
        metrics["images"] = len(selected)
        for key in (
            "eligible_gaps",
            "closed_gaps",
            "gap_closure_rate",
            "background_bridge_proxy",
            "new_tp",
            "new_fp",
            "isolated_new_fp_pixels",
        ):
            metrics[key] = float(np.mean([row[key] for row in selected]))
        mode_summary[name] = metrics

    def delta(mode, field):
        return mode_summary[mode][field] - mode_summary["baseline"][field]

    contrast_readout = {
        "learned_w_vs_uniform": {
            key: -delta("uniform_w", key)
            for key in (
                "iou",
                "cldice",
                "apls_gt_to_pred_approx",
                "apls_bidirectional_approx",
                "gap_closure_rate",
                "background_bridge_proxy",
            )
        },
        "learned_w_vs_rotated90": {
            key: -delta("rotated90_w", key)
            for key in (
                "iou",
                "cldice",
                "apls_gt_to_pred_approx",
                "apls_bidirectional_approx",
                "gap_closure_rate",
                "background_bridge_proxy",
            )
        },
        "frozen_sources_vs_dynamic": {
            "new_tp_delta": delta("frozen_r_sources", "new_tp"),
            "new_fp_delta": delta("frozen_r_sources", "new_fp"),
            "isolated_new_fp_delta": delta(
                "frozen_r_sources", "isolated_new_fp_pixels"
            ),
            "gap_closure_delta": delta("frozen_r_sources", "gap_closure_rate"),
            "bridge_proxy_delta": delta("frozen_r_sources", "background_bridge_proxy"),
        },
        "oracle_r_vs_dynamic": {
            "new_tp_delta": delta("oracle_clear_r_background", "new_tp"),
            "new_fp_delta": delta("oracle_clear_r_background", "new_fp"),
            "isolated_new_fp_delta": delta(
                "oracle_clear_r_background", "isolated_new_fp_pixels"
            ),
            "gap_closure_delta": delta("oracle_clear_r_background", "gap_closure_rate"),
            "bridge_proxy_delta": delta(
                "oracle_clear_r_background", "background_bridge_proxy"
            ),
        },
        "oracle_g_vs_dynamic": {
            "new_tp_delta": delta("oracle_g_background", "new_tp"),
            "new_fp_delta": delta("oracle_g_background", "new_fp"),
            "isolated_new_fp_delta": delta(
                "oracle_g_background", "isolated_new_fp_pixels"
            ),
            "gap_closure_delta": delta("oracle_g_background", "gap_closure_rate"),
            "bridge_proxy_delta": delta(
                "oracle_g_background", "background_bridge_proxy"
            ),
        },
    }
    direction_summary = {}
    for region in ("recovered_gt", "unrecovered_gt", "reliable_road"):
        selected = [
            row
            for row in all_directions
            if row.get("region") == region and row.get("valid_work_sites", 0) > 0
        ]
        site_total = sum(row["valid_work_sites"] for row in selected)
        direction_summary[region] = {"valid_work_sites": site_total}
        for key in (
            "axis_consistency",
            "weighted_angle_error_deg",
            "transport_weighted_axis_consistency",
            "w_entropy_mean",
            "w_max_mean",
            "alpha_mean",
        ):
            values = [row for row in selected if row.get(key) is not None]
            direction_summary[region][key] = (
                float(
                    sum(row[key] * row["valid_work_sites"] for row in values)
                    / max(sum(row["valid_work_sites"] for row in values), 1)
                )
                if values
                else None
            )
        for key in ("angle_error_p50_deg", "angle_error_p90_deg"):
            values = [row for row in selected if row.get(key) is not None]
            direction_summary[region][f"mean_tile_{key}"] = (
                float(np.mean([row[key] for row in values])) if values else None
            )
        direction_summary[region]["owned_tiles"] = len(selected)
    saturation_summary = {}
    for region in (
        "recovered_gt",
        "unrecovered_gt",
        "isolated_new_fp",
        "false_bridge",
        "hard_background",
    ):
        selected = [row for row in all_saturation if row["region"] == region]
        saturation_summary[region] = {}
        for key in (
            "delta_plus_ge_7_9",
            "delta_plus_ge_7_99",
            "delta_plus_ge_7_999",
            "delta_plus_at_8",
            "plus_derivative_mean",
            "plus_derivative_zero_fraction",
            "plus_loss_gradient_mean_abs",
            "plus_loss_gradient_zero_fraction",
            "raw_plus_p50",
            "delta_minus_mean",
            "minus_near_zero_fraction",
            "minus_derivative_mean",
            "minus_loss_gradient_mean_abs",
            "minus_loss_gradient_zero_fraction",
        ):
            values = [row for row in selected if row.get(key) is not None]
            if key.endswith("gradient_mean_abs"):
                saturation_summary[region][key] = (
                    float(np.mean([row[key] for row in values])) if values else None
                )
            else:
                saturation_summary[region][key] = (
                    float(
                        sum(row[key] * row["work_sites"] for row in values)
                        / max(sum(row["work_sites"] for row in values), 1)
                    )
                    if values
                    else None
                )
    source_dynamics_summary = []
    for mode in (
        "baseline",
        "frozen_r_sources",
        "oracle_clear_r_background",
        "oracle_g_background",
    ):
        for region in (
            "recovered_gt",
            "unrecovered_gt",
            "isolated_new_fp",
            "false_bridge",
        ):
            selected = [
                row
                for row in all_source_modes
                if row["mode"] == mode and row["region"] == region
            ]
            summary_row = {
                "mode": mode,
                "region": region,
                "owned_tiles": len(selected),
                "work_pixels": sum(row["work_pixels"] for row in selected),
            }
            for key in (
                "v1_mean",
                "v2_mean",
                "v3_mean",
                "alpha1_mean",
                "alpha2_mean",
                "alpha3_mean",
                "transport_delta1_abs",
                "transport_delta2_abs",
                "transport_delta3_abs",
            ):
                values = [
                    row
                    for row in selected
                    if row.get(key) is not None and row["work_pixels"] > 0
                ]
                summary_row[key] = (
                    float(
                        sum(row[key] * row["work_pixels"] for row in values)
                        / max(sum(row["work_pixels"] for row in values), 1)
                    )
                    if values
                    else None
                )
            source_dynamics_summary.append(summary_row)
    gate_summary = {}
    for region in ("recovered_gt", "unrecovered_gt", "isolated_new_fp", "false_bridge"):
        selected = [row for row in all_gate_rows if row["region"] == region]
        gate_summary[region] = {
            f"{field}_{quantile}_mean_of_owned_tile_quantiles": (
                float(
                    np.mean(
                        [
                            row[f"{field}_{quantile}"]
                            for row in selected
                            if row[f"{field}_{quantile}"] is not None
                        ]
                    )
                )
                if any(row[f"{field}_{quantile}"] is not None for row in selected)
                else None
            )
            for field in ("g_transport", "c_bridge")
            for quantile in ("p10", "p50", "p90")
        }
    w_directional_advantage = (
        contrast_readout["learned_w_vs_uniform"]["apls_gt_to_pred_approx"] > 0
        and contrast_readout["learned_w_vs_rotated90"]["apls_gt_to_pred_approx"] > 0
        and contrast_readout["learned_w_vs_uniform"]["gap_closure_rate"] >= 0
        and contrast_readout["learned_w_vs_rotated90"]["gap_closure_rate"] >= 0
    )
    source_modes = (
        "frozen_sources_vs_dynamic",
        "oracle_r_vs_dynamic",
        "oracle_g_vs_dynamic",
    )
    strongest_source_intervention = min(
        source_modes,
        key=lambda name: contrast_readout[name]["new_fp_delta"],
    )
    conclusion_lines = [
        (
            "W direction: learned W beats both uniform and 90-degree-rotated W on "
            "GT→prediction APLS without reducing closure on this paired validation set."
            if w_directional_advantage
            else "W direction: the paired results do not establish that learned W is "
            "better than both uniform and 90-degree-rotated W on GT→prediction APLS "
            "and gap closure."
        ),
        (
            f"R/source versus transport permission: {strongest_source_intervention} "
            "produced the largest reduction in added FP among the tested interventions; "
            "interpret alongside its TP and gap-closure deltas before attributing cause."
        ),
    ]
    report = {
        "checkpoint": str(args.dynamic_checkpoint),
        "split": "val",
        "images": args.images,
        "threshold": args.threshold,
        "direction_order_dy_dx": [list(value) for value in DIRECTIONS],
        "direction_semantics": "each channel reads source at target p+(dy,dx); axial comparisons treat opposite vectors as the same road tangent",
        "counterfactual_metrics": mode_summary,
        "paired_contrast_deltas_vs_baseline": contrast_readout,
        "direction_alignment_summary": direction_summary,
        "delta_saturation_summary": saturation_summary,
        "source_dynamics_summary": source_dynamics_summary,
        "gate_region_summary": gate_summary,
        "evidence_limited_conclusions_max_2": conclusion_lines,
        "matched_connection_pair_ranking": matched,
        "connection_pair_method": "positive pairs are flanks of connected GT skeleton segments missing from P0; negative pairs are nearby endpoints from distinct GT 8-connected components. Scores are aggregated along the GT missing skeleton pixels or straight negative corridor and matched by endpoint P0/Q3/R bins and endpoint distance bins.",
        "connection_pair_generation": {
            "reference_missing_segments_ge3px": sum(
                row["reference_missing_segments_ge3px"] for row in all_pair_summaries
            ),
            "positive_pairs_generated": sum(
                row["positive_pairs_generated"] for row in all_pair_summaries
            ),
            "positive_pairs_rejected": sum(
                row["positive_pairs_rejected"] for row in all_pair_summaries
            ),
            "positive_pair_generation_recall": (
                sum(row["positive_pairs_generated"] for row in all_pair_summaries)
                / max(
                    sum(
                        row["reference_missing_segments_ge3px"]
                        for row in all_pair_summaries
                    ),
                    1,
                )
            ),
            "negative_pairs_generated": sum(
                row["negative_pairs_generated"] for row in all_pair_summaries
            ),
            "matched_positive_pairs": sum(
                row["matched_positive_pairs"] for row in all_pair_summaries
            ),
            "matched_negative_pairs": sum(
                row["matched_negative_pairs"] for row in all_pair_summaries
            ),
        },
        "source_formula": "SourceSupport_k = R + (1-R)*clamp((U_k-P0)_+/(1-P0),0,1); V_k=clamp(sum_d W_dir[d]*shift(SourceSupport_k,d),0,1); alpha_k=softmax_rate_transport_k*G_transport. Frozen-source mode sets SourceSupport_k=R for all steps. R oracle sets R=R*GT_work and recomputes relation/proposal/acceptance.",
        "coefficient_explanation": "In this v6 code alpha=transport_rate*G_transport and beta=0.2*A*proposal_rate. A does not multiply alpha, so alpha can exceed A; V enters the rate network's features but is not a multiplicative factor on alpha.",
        "gate_formula": "G_transport=sigmoid(direction_gate_logit+log(max_directional_relation_evidence)); W_dir=softmax(direction_logits+log(directional_relation_evidence+0.05)).",
        "g_oracle": "G_transport is multiplied by GT_work at the receiving pixel; diagnostic-only oracle.",
        "delta_plus_formula": "DeltaZ_plus=8*sigmoid(raw_plus), smoothly bounded below 8 (not hard-clipped); derivative=8*sigma*(1-sigma). DeltaZ_minus=4*sigmoid(raw_minus).",
        "limitations": [
            "Direction, source-trace, gate and saturation statistics use unique owned work-grid sites from every 512/256 tile and regions are labeled against stitched P0/P1. Counts are work-grid sites, not original-resolution pixels.",
            "Source-trace CSV samples up to 100 highest positive-transport false-positive target sites per tile; it does not enumerate every changed pixel.",
            "Proposal-loss gradients are measured on the most gap-rich tile per image only; saturation/value distributions cover all owned tile regions.",
            "Connection pairs are diagnostic GT-derived candidates, not deployable predictions. Positive generation requires a two-ended missing skeleton component and matched flanks; rejected/unsupported segments are counted.",
            "Negative candidates are nearby endpoints from distinct GT connected components, scored along their straight corridor; they are hard negatives but do not cover every possible false connection geometry.",
            "Pair ranking only uses common endpoint-context/distance strata; unmatched pairs are excluded and pair generation recall must be consulted.",
            "Oracle modes are diagnostic-only and not deployable.",
            "This script does not alter checkpoints or optimizer state.",
        ],
    }
    (args.output_dir / "causal_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    report_md = [
        "# v6 causal contrasts",
        "",
        f"- Checkpoint: `{args.dynamic_checkpoint}`",
        f"- Validation images: {args.images}; threshold: {args.threshold:.2f}",
        "- No model weights, losses, or optimizer state were changed.",
        "",
        "## Counterfactual performance",
        "",
        "| Mode | IoU | clDice | GT→pred APLS | Bidirectional APLS | gap closure | bridge proxy | new TP | new FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in mode_summary.items():
        report_md.append(
            f"| {name} | {metrics.get('iou', 0):.4f} | {metrics.get('cldice', 0):.4f} | {metrics.get('apls_gt_to_pred_approx', 0):.4f} | {metrics.get('apls_bidirectional_approx', 0):.4f} | {metrics.get('gap_closure_rate', 0):.4f} | {metrics.get('background_bridge_proxy', 0):.2f} | {metrics.get('new_tp', 0):.1f} | {metrics.get('new_fp', 0):.1f} |"
        )
    report_md += [
        "",
        "## Matched connection-pair ranking",
        "",
        "| Path score | ROC-AUC | AP | matched positive/negative |",
        "|---|---:|---:|---:|",
    ]
    for name, metrics in matched.items():
        report_md.append(
            f"| {name} | {metrics['roc_auc']} | {metrics['average_precision']} | {metrics['positive_count']}/{metrics['negative_count']} |"
        )
    report_md += [
        "",
        "Connection-pair generation summary: "
        + json.dumps(report["connection_pair_generation"], sort_keys=True),
    ]
    report_md += [
        "",
        "## Paired causal contrasts",
        "",
        "Deltas below are intervention minus the original-W dynamic output; positive/negative signs are literal metric changes.",
        "",
        "```json",
        json.dumps(contrast_readout, indent=2),
        "```",
    ]
    report_md += [
        "",
        "## W versus GT tangent",
        "",
        "| Region | Owned tiles | Valid work sites | Axis consistency | Weighted angle error | Transport-weighted consistency | Entropy | W max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region, values in direction_summary.items():
        report_md.append(
            f"| {region} | {values['owned_tiles']} | {values['valid_work_sites']} | {values['axis_consistency']} | {values['weighted_angle_error_deg']} | {values['transport_weighted_axis_consistency']} | {values['w_entropy_mean']} | {values['w_max_mean']} |"
        )
    report_md += [
        "",
        "## ΔZ proposal saturation",
        "",
        "| Region | ΔZ+ ≥7.9 | ≥7.99 | ≥7.999 | =8 | dΔZ+/draw | loss grad + | ΔZ− mean | ΔZ− <0.01 | loss grad − |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region, values in saturation_summary.items():
        report_md.append(
            f"| {region} | {values['delta_plus_ge_7_9']} | {values['delta_plus_ge_7_99']} | {values['delta_plus_ge_7_999']} | {values['delta_plus_at_8']} | {values['plus_derivative_mean']} | {values['plus_loss_gradient_mean_abs']} | {values['delta_minus_mean']} | {values['minus_near_zero_fraction']} | {values['minus_loss_gradient_mean_abs']} |"
        )
    report_md += ["", "## Evidence-limited conclusions (maximum 2)", ""]
    report_md.extend(f"- {line}" for line in conclusion_lines)
    report_md += [
        "",
        "## R source and G permission interventions",
        "",
        "| Region | Mode | V1 | V2 | V3 | α1 | α2 | α3 | |transport Δ1| | Δ2 | Δ3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in source_dynamics_summary:
        report_md.append(
            f"| {row['region']} | {row['mode']} | {row['v1_mean']} | {row['v2_mean']} | {row['v3_mean']} | {row['alpha1_mean']} | {row['alpha2_mean']} | {row['alpha3_mean']} | {row['transport_delta1_abs']} | {row['transport_delta2_abs']} | {row['transport_delta3_abs']} |"
        )
    report_md += [
        "",
        "## G_transport and C_bridge by error class",
        "",
        "| Region | G mean tile p10/p50/p90 | C_bridge mean tile p10/p50/p90 |",
        "|---|---:|---:|",
    ]
    for region, values in gate_summary.items():
        g = "/".join(
            str(values[f"g_transport_{quantile}_mean_of_owned_tile_quantiles"])
            for quantile in ("p10", "p50", "p90")
        )
        c = "/".join(
            str(values[f"c_bridge_{quantile}_mean_of_owned_tile_quantiles"])
            for quantile in ("p10", "p50", "p90")
        )
        report_md.append(f"| {region} | {g} | {c} |")
    report_md += [
        "",
        "## Formula and interpretation",
        "",
        report["source_formula"],
        "",
        report["coefficient_explanation"],
        "",
        report["gate_formula"],
        "",
        report["delta_plus_formula"],
        "",
        "## Caveats",
        "",
    ]
    report_md.extend(f"- {item}" for item in report["limitations"])
    (args.output_dir / "v6_causal_report.md").write_text(
        "\n".join(report_md) + "\n", encoding="utf-8"
    )
    print(f"[V6 CAUSAL] report: {args.output_dir / 'v6_causal_report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", type=Path, required=True)
    parser.add_argument("--dynamic_checkpoint", type=Path, required=True)
    parser.add_argument("--base_checkpoint", type=Path)
    parser.add_argument("--q3_checkpoint", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--images", type=int, default=14)
    parser.add_argument("--threshold", type=float, default=0.45)
    args = parser.parse_args()
    if args.images < 1 or not 0 < args.threshold < 1:
        parser.error("images must be positive and threshold must be between 0 and 1")
    if not args.root_path.is_dir():
        parser.error(f"validation dataset directory does not exist: {args.root_path}")
    if not args.dynamic_checkpoint.is_file():
        parser.error(
            "--dynamic_checkpoint must point to an existing checkpoint file; "
            f"got {args.dynamic_checkpoint}"
        )
    run(args)


if __name__ == "__main__":
    main()
