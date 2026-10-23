"""Paired diagnostics for encoder evidence in H0 local-gap candidate discovery.

Validation 1 evaluates fixed P2-only proposals against proposals augmented by
frozen E2/H_E2/H_E3/E1/H0/F1 probe evidence. Validation 2 trains matched
candidate scorers on frozen path features and reports validation ranking and
coverage. It never updates either backbone or selects a test threshold.

Run from the local-gap diagnostics checkout. --probe_repo must point to the
exact a4001b3 E2-direction code checkout; the heads must match its checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import _C
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnose_gap_candidates import load_model as load_gap_model
from gap_candidates import endpoints as extract_endpoints
from gap_candidates import generate_candidates
from local_gap_config import LocalGapConfig
from losses.gap_losses import label_edges, proposal_gap_coverage
from mass_protocol import sliding_positions
from evaluation.road_comparison.road_metrics import skeletonize


PROBE_NAMES = ("E2", "H_E2", "H_E3", "E1", "H0", "F1")
FEATURE_LAYOUT = {"F2": 96, "H_E2": 64, "H_E3": 64,
                  "E1": 96, "H0": 64, "F1": 192}
FEATURE_DIMS = tuple(FEATURE_LAYOUT.values())
SCORER_FEATURES = {
    "R0_F2": ("F2",),
    "R1_F2_HE2": ("F2", "H_E2"),
    "R2_F2_HE2_HE3": ("F2", "H_E2", "H_E3"),
    "R3_F2_E1": ("F2", "E1"),
    "R4_F2_H0": ("F2", "H0"),
    "R5_F2_F1": ("F2", "F1"),
}
GEOMETRY_DIMS = 8
FEATURE_DIM = sum(FEATURE_DIMS) + GEOMETRY_DIMS


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def load_probe_stack(probe_repo: Path, backbone_path: Path, probes_path: Path, device):
    """Load the strict a400 encoder checkpoint after the 39e Gap model is built."""
    probe_repo = probe_repo.resolve()
    if not (probe_repo / "h0_e2_config.py").is_file():
        raise FileNotFoundError(f"a4001b3 code missing under --probe_repo {probe_repo}")
    # Both source trees use top-level packages named networks/config/datasets.
    # The already-created Gap model retains its module objects; clear import
    # cache entries before importing the second, encoder-exchange source tree.
    for name in list(sys.modules):
        if (name == "a400_probe_experiment" or name == "h0_e2_config" or
                name == "h0_encoder_protocol" or
                name == "networks" or name.startswith("networks.") or
                name == "config" or name.startswith("config.") or
                name == "datasets" or name.startswith("datasets.") or
                name == "losses" or name.startswith("losses.")):
            sys.modules.pop(name, None)
    sys.path.insert(0, str(probe_repo))
    try:
        exp = importlib.import_module("a400_probe_experiment")
        backbone = exp.load_backbone(str(backbone_path), device)
        backbone_hash = exp.file_sha256(backbone_path)
        heads, payload = exp.load_heads(probes_path, backbone_hash, device)
        tap = exp.FeatureTap(backbone)
        return exp, backbone, heads, tap, payload
    finally:
        # Keep the encoder source at the front while its modules are live.
        # The H0 model already holds references to its original source modules.
        pass


def endpoint_sources(p2: np.ndarray, evidence: np.ndarray | None, cfg: LocalGapConfig,
                     evidence_threshold: float):
    """Return tagged endpoints; P2 endpoints remain the preferred strong anchors."""
    p2_points, p2_raw = extract_endpoints(p2, cfg)
    tagged = [(point, tangent, score, "p2") for point, tangent, score in p2_points]
    if evidence is not None:
        ev_cfg = replace(cfg, threshold=evidence_threshold)
        ev_points, _ = extract_endpoints(evidence, ev_cfg)
        occupied = {tuple(item[0]) for item in tagged}
        for point, tangent, score in ev_points:
            key = tuple(point)
            if key not in occupied:
                tagged.append((point, tangent, score, "evidence"))
                occupied.add(key)
    # Keep every P2 anchor first; add strongest evidence endpoints within the
    # same total endpoint budget used by the original generator.
    strong = [item for item in tagged if item[3] == "p2"]
    weak = sorted((item for item in tagged if item[3] == "evidence"),
                  key=lambda item: (-item[2], tuple(item[0])))
    tagged = (strong + weak)[:cfg.max_endpoints]
    return tagged, p2_raw


def line_mean(probability: np.ndarray, u, v, samples=16):
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    points = np.asarray(u, np.float32)[None] * (1 - t[:, None]) + np.asarray(v, np.float32)[None] * t[:, None]
    y = np.clip(np.rint(points[:, 0]).astype(int), 0, probability.shape[0] - 1)
    x = np.clip(np.rint(points[:, 1]).astype(int), 0, probability.shape[1] - 1)
    return float(probability[y, x].mean())


def expanded_candidates(p2: np.ndarray, evidence: np.ndarray, cfg: LocalGapConfig,
                        evidence_threshold: float, candidate_pool: int):
    """Evidence adds endpoint proposals and ranks weak corridors before the budget cap."""
    points, raw_count = endpoint_sources(p2, evidence, cfg, evidence_threshold)
    # Associate every proposal endpoint with a P2 connected component. A
    # candidate must join two distinct P2-supported road pieces.
    p2_skeleton = skeletonize(p2 >= cfg.threshold).astype(np.uint8)
    _, p2_labels = cv2.connectedComponents(p2_skeleton, connectivity=8)
    tagged_labels = []
    max_snap = 3.0  # pixels on the P2/128 grid, held fixed for all variants
    coords = np.argwhere(p2_skeleton > 0)
    for point, tangent, score, source in points:
        y, x = point
        label = int(p2_labels[y, x]) if 0 <= y < p2_labels.shape[0] and 0 <= x < p2_labels.shape[1] else 0
        if label == 0 and len(coords):
            distances = np.linalg.norm(coords.astype(np.float32) - np.asarray(point, np.float32), axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= max_snap:
                label = int(p2_labels[tuple(coords[nearest])])
        tagged_labels.append(label)

    candidates = []
    for i, (u, tu, pu, source_u) in enumerate(points):
        for j in range(i + 1, len(points)):
            v, tv, pv, source_v = points[j]
            if tagged_labels[i] == 0 or tagged_labels[j] == 0 or tagged_labels[i] == tagged_labels[j]:
                continue
            delta = np.asarray(v, np.float32) - np.asarray(u, np.float32)
            distance = float(np.linalg.norm(delta))
            if not cfg.min_distance <= distance <= cfg.max_distance:
                continue
            unit = delta / max(distance, 1e-6)
            cu, cv = float(np.asarray(tu) @ unit), float(np.asarray(tv) @ -unit)
            # Do not hard-reject weakly aligned pairs: probe support may recover
            # a weak/missing endpoint direction. Rank them instead.
            corridor = line_mean(evidence, u, v)
            p2_corridor = line_mean(p2, u, v)
            geometry = [distance / cfg.max_distance, cu, cv, float(np.asarray(tu) @ np.asarray(tv)),
                        float(unit[1]), float(unit[0]), (float(pu) + float(pv)) / 2,
                        min(float(pu), float(pv))]
            score = (0.55 * corridor + 0.25 * min(cu, cv) +
                     0.10 * p2_corridor - 0.15 * distance / cfg.max_distance)
            # Stable coordinates make ordering deterministic across runs.
            candidates.append((score, tuple(map(int, u)), tuple(map(int, v)), geometry,
                               source_u, source_v, corridor))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected, seen = [], set()
    for item in candidates:
        key = tuple(sorted((item[1], item[2])))
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) >= candidate_pool:
            break
    edges = np.asarray([[item[1], item[2]] for item in selected], dtype=np.float32).reshape(-1, 2, 2)
    geometry = np.asarray([item[3] for item in selected], dtype=np.float32).reshape(-1, 8)
    scores = np.asarray([item[0] for item in selected], dtype=np.float32)
    records = [{"score": float(item[0]), "source_u": item[4], "source_v": item[5],
                "corridor_evidence": float(item[6])} for item in selected]
    return {"edges": edges, "geometry": geometry, "scores": scores,
            "records": records, "raw_endpoint_count": raw_count,
            "retained_endpoint_count": len(points), "eligible_pairs": len(candidates)}


def pooled_path_feature(feature: np.ndarray, edge, p2_hw):
    """Channel means sampled along a candidate; feature is CHW, edge is P2 yx."""
    channels, height, width = feature.shape
    t = np.linspace(0, 1, 12, dtype=np.float32)
    points = np.asarray(edge[0], np.float32)[None] * (1 - t[:, None]) + np.asarray(edge[1], np.float32)[None] * t[:, None]
    ys = np.clip(np.rint(points[:, 0] * height / p2_hw[0]).astype(int), 0, height - 1)
    xs = np.clip(np.rint(points[:, 1] * width / p2_hw[1]).astype(int), 0, width - 1)
    return feature[:, ys, xs].mean(axis=1, dtype=np.float32)


def build_edge_features(feature_maps, edge, geometry, p2_hw):
    pieces = []
    for name, channels in FEATURE_LAYOUT.items():
        value = feature_maps.get(name)
        if value is None:
            pooled = np.zeros(channels, np.float32)
        else:
            pooled = pooled_path_feature(value, edge, p2_hw)
            if pooled.size != channels:
                raise ValueError(f"{name}: expected {channels} channels, got {pooled.size}")
        pieces.append(pooled)
    return np.concatenate([*pieces, np.asarray(geometry, np.float32)])


def edge_labels(mask, skeleton, edges, cfg):
    return label_edges(skeleton.astype(bool), mask.astype(bool), edges,
                       (128, 128), cfg)


def average_precision(y, score):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    valid = y >= 0
    y, score = y[valid], score[valid]
    if not len(y) or not np.any(y == 1):
        return None
    order = np.argsort(-score, kind="stable")
    positive = (y[order] == 1).astype(np.float64)
    return float((positive * (positive.cumsum() / np.arange(1, len(y) + 1))).sum() / positive.sum())


class CandidateScorer(nn.Module):
    """Identical parameter count for every R variant; excluded blocks are zeroed."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURE_DIM, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def fit_scorer(train_x, train_y, seed, device, epochs=15):
    valid = np.asarray(train_y) >= 0
    x = torch.as_tensor(np.asarray(train_x)[valid], dtype=torch.float32, device=device)
    y = torch.as_tensor(np.asarray(train_y)[valid], dtype=torch.float32, device=device)
    if len(y) < 8 or not (y == 1).any() or not (y == 0).any():
        raise RuntimeError(f"Candidate-head training needs both classes; got {len(y)} labels")
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    permutation = rng.permutation(len(y))
    model = CandidateScorer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.clamp((y == 0).sum().float() / (y == 1).sum().clamp_min(1), 1.0, 20.0)
    history = []
    model.train()
    for epoch in range(epochs):
        rng.shuffle(permutation)
        losses = []
        for start in range(0, len(permutation), 512):
            index = permutation[start:start + 512]
            logits = model(x[index])
            loss = F.binary_cross_entropy_with_logits(logits, y[index], pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    model.eval()
    return model, history


def enabled_input(x, variant):
    result = x.copy()
    if variant not in SCORER_FEATURES:
        raise ValueError(variant)
    offset = 0
    for name, channels in FEATURE_LAYOUT.items():
        if name not in SCORER_FEATURES[variant]:
            result[..., offset:offset + channels] = 0
        offset += channels
    return result


def make_feature_tap_gap(model):
    captured = {}
    handles = []
    def gap_hook(_module, inputs, outputs):
        captured["p2"] = outputs[1]["p2_logits"].detach()
    def f2_hook(_module, _inputs, outputs):
        value = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if not torch.is_tensor(value) or value.ndim != 4:
            raise RuntimeError("Could not capture the post-Stage-2 decoder feature F2")
        captured["f2"] = value.detach()
    handles.append(model.swin_unet.gap_connection.register_forward_hook(gap_hook))
    handles.append(model.swin_unet.decoder_structure_blocks["2"].register_forward_hook(f2_hook))
    return captured, handles


def extract_probe_probabilities(exp, heads, features, size=(128, 128)):
    output = {}
    for name in PROBE_NAMES:
        logits = exp.probe_logits(heads[name], features[name])
        probability = logits.sigmoid()
        probability = F.interpolate(probability, size=size, mode="bilinear", align_corners=False)
        output[name] = probability[0, 0].float().cpu().numpy()
    return output


def to_numpy_feature(value):
    return value[0].float().cpu().numpy()


def tile_iter(dataset, split, max_images, seed):
    count = len(dataset) if max_images <= 0 else min(len(dataset), max_images)
    for image_index in range(count):
        sample = dataset[image_index]
        image = sample["image"][None]
        mask = sample["mask"][0].numpy() > .5
        skeleton = sample["skeleton"][0].numpy() > .5
        name = Path(sample["image_name"]).stem
        if split == "train":
            rng = np.random.RandomState(seed + image_index * 9176)
            top = int(rng.randint(0, 1500 - 512 + 1))
            left = int(rng.randint(0, 1500 - 512 + 1))
            positions = [(top, left)]
        else:
            positions = [(top, left) for top in sliding_positions(1500, 512, 256)
                         for left in sliding_positions(1500, 512, 256)]
        for tile_index, (top, left) in enumerate(positions):
            yield {"image": image[:, :, top:top + 512, left:left + 512],
                   "mask": mask[top:top + 512, left:left + 512],
                   "skeleton": skeleton[top:top + 512, left:left + 512],
                   "key": f"{name}:{top}:{left}", "image_name": name,
                   "top": top, "left": left, "tile_index": tile_index}


def process_tile(tile, gap_model, encoder_tap, heads, exp, capture, cfg,
                 candidate_pool, candidate_budget, evidence_threshold, device):
    image = tile["image"].to(device)
    capture.clear()
    with torch.inference_mode():
        gap_model(image)
        if "p2" not in capture:
            raise RuntimeError("Gap model did not expose P2 logits")
        p2 = capture["p2"][0, 0].float().sigmoid().cpu().numpy()
        f2 = to_numpy_feature(capture["f2"])
        features, _ = encoder_tap(image)
        probe_probs = extract_probe_probabilities(exp, heads, features)
    feature_maps = {"F2": f2, **{name: to_numpy_feature(features[name])
                                 for name in FEATURE_LAYOUT if name != "F2"}}
    variants = {"P0_P2_only": generate_candidates([p2], cfg)[0]}
    evidence_sets = {
        "P1_P2_plus_E2": ("E2",),
        "P2_P2_plus_HE2": ("H_E2",),
        "P3_P2_plus_HE3": ("H_E3",),
        "P4_P2_plus_HE2_HE3": ("H_E2", "H_E3"),
        "P5_P2_plus_E1": ("E1",),
        "P6_P2_plus_H0": ("H0",),
        "P7_P2_plus_F1": ("F1",),
    }
    expanded = {}
    for name, sources in evidence_sets.items():
        evidence = np.maximum.reduce([probe_probs[source] for source in sources])
        proposal = expanded_candidates(p2, evidence, cfg, evidence_threshold, candidate_pool)
        expanded[name] = proposal
        variants[name] = {"edges": proposal["edges"][:candidate_budget],
                          "geometry": proposal["geometry"][:candidate_budget]}
    # The common pool for validation 2 is the coordinate-deduplicated union of
    # all candidates. Every scorer receives exactly the same examples.
    union = {}
    candidate_sets = [variants["P0_P2_only"], *[expanded[name] for name in evidence_sets]]
    max_count = max((len(candidate_set["edges"]) for candidate_set in candidate_sets), default=0)
    # Round-robin prevents the first evidence source from consuming the whole
    # shared pool before H_E2/H_E3 candidates are represented.
    for rank in range(max_count):
        for candidate_set in candidate_sets:
            if rank >= len(candidate_set["edges"]):
                continue
            edge, geometry = candidate_set["edges"][rank], candidate_set["geometry"][rank]
            key = tuple(np.round(edge.reshape(-1), 3))
            union.setdefault(key, (edge, geometry))
            if len(union) >= candidate_pool:
                break
        if len(union) >= candidate_pool:
            break
    union_items = list(union.values())
    union_edges = np.asarray([item[0] for item in union_items], np.float32).reshape(-1, 2, 2)
    union_geometry = np.asarray([item[1] for item in union_items], np.float32).reshape(-1, 8)
    descriptors = np.asarray([build_edge_features(feature_maps, edge, geom, p2.shape)
                              for edge, geom in zip(union_edges, union_geometry)], np.float32).reshape(-1, FEATURE_DIM)
    labels, reasons = label_edges(tile["skeleton"], tile["mask"], union_edges, p2.shape, cfg)
    compact_tile = {key: value for key, value in tile.items() if key != "image"}
    return {"tile": compact_tile, "p2": p2,
            "variants": variants, "expanded": expanded, "union_edges": union_edges,
            "union_geometry": union_geometry, "descriptors": descriptors,
            "union_labels": labels, "label_reasons": reasons}


def proposal_rows(record, cfg, candidate_budget):
    tile, p2 = record["tile"], record["p2"]
    rows, totals = [], {}
    variants = record["variants"]
    for name, proposal in variants.items():
        edges = proposal["edges"][:candidate_budget]
        geometry = proposal["geometry"][:candidate_budget]
        labels, reasons = label_edges(tile["skeleton"], tile["mask"], edges, p2.shape, cfg)
        verified, covered = proposal_gap_coverage(tile["skeleton"], p2, edges, labels, cfg)
        counts = Counter(map(int, labels.tolist()))
        reason_counts = Counter(reasons)
        row = {"tile": tile["key"], "variant": name, "candidates": len(edges),
               "verified_gaps": int(verified), "covered_gaps": int(covered),
               "gap_recall": float(covered / verified) if verified else None,
               "positive_edges": counts[1], "clear_negative_edges": counts[0],
               "ambiguous_edges": counts[-1],
               "no_local_path_negatives": reason_counts["no_local_path"],
               "background_endpoint_negatives": reason_counts["background_endpoint"]}
        rows.append(row)
        slot = totals.setdefault(name, Counter())
        for key in ("candidates", "verified_gaps", "covered_gaps", "positive_edges",
                    "clear_negative_edges", "ambiguous_edges", "no_local_path_negatives",
                    "background_endpoint_negatives"):
            slot[key] += int(row[key] or 0)
    return rows, totals


def score_validation_heads(train_records, val_records, cfg, device, seed, epochs, candidate_budget):
    variants = tuple(SCORER_FEATURES)
    nonempty_train = [record for record in train_records if len(record["descriptors"])]
    if not nonempty_train:
        raise RuntimeError("No training candidates were generated; candidate scorers cannot be trained")
    train_x = np.concatenate([record["descriptors"] for record in nonempty_train])
    train_y = np.concatenate([record["union_labels"] for record in nonempty_train])
    if train_x.shape != (len(train_y), FEATURE_DIM):
        raise RuntimeError(f"Training candidate feature shape mismatch: {train_x.shape}, {train_y.shape}")
    # Standardize once from the training partition only; this preserves a
    # fair, identical feature scale across all scorers.
    valid = train_y >= 0
    if not valid.any():
        raise RuntimeError("All training candidates have ambiguous GT labels")
    mean = train_x[valid].mean(0)
    std = train_x[valid].std(0).clip(1e-4)
    results, history_all = {}, {}
    for variant in variants:
        x = enabled_input((train_x - mean) / std, variant)
        model, history = fit_scorer(x, train_y, seed, device, epochs)
        history_all[variant] = history
        val_scores, val_labels = [], []
        gap_total = gap_covered = 0
        candidate_counts = positive_counts = negative_counts = 0
        with torch.no_grad():
            for record in val_records:
                raw = record["descriptors"]
                if raw.shape != (len(record["union_labels"]), FEATURE_DIM):
                    raise RuntimeError(f"Validation candidate feature shape mismatch: {raw.shape}")
                xval = enabled_input((raw - mean) / std, variant)
                logits = model(torch.as_tensor(xval, device=device)).cpu().numpy() if len(xval) else np.zeros(0)
                order = np.argsort(-logits, kind="stable")[:candidate_budget]
                edges = record["union_edges"][order]
                labels = record["union_labels"][order]
                verified, covered = proposal_gap_coverage(
                    record["tile"]["skeleton"], record["p2"], edges, labels, cfg)
                gap_total += int(verified); gap_covered += int(covered)
                candidate_counts += len(edges)
                positive_counts += int((labels == 1).sum())
                negative_counts += int((labels == 0).sum())
                # AP assesses ranking over the complete common pool; the fixed
                # budget is applied separately to the gap-recall/precision stats.
                val_scores.extend(logits.tolist())
                val_labels.extend(record["union_labels"].tolist())
        results[variant] = {"candidate_budget_per_tile": candidate_budget,
                            "candidates_evaluated": candidate_counts,
                            "positive_edges_at_budget": positive_counts,
                            "clear_negatives_at_budget": negative_counts,
                            "verified_gaps": gap_total, "covered_gaps": gap_covered,
                            "gap_recall_at_budget": gap_covered / gap_total if gap_total else None,
                            "edge_average_precision": average_precision(val_labels, val_scores)}
    return results, history_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--gap_model_path", required=True, help="39e1398 local-gap checkpoint")
    parser.add_argument("--probe_repo", required=True, help="checkout of a4001b3 source")
    parser.add_argument("--probe_model_path", required=True, help="a4001b3 E2-direction checkpoint")
    parser.add_argument("--probe_head_path", required=True, help="matching ten-probe probe_best.pth")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_val_images", type=int, default=0)
    parser.add_argument("--max_train_images", type=int, default=256)
    parser.add_argument("--candidate_pool", type=int, default=256)
    parser.add_argument("--candidate_budget", type=int, default=64)
    parser.add_argument("--evidence_threshold", type=float, default=.45)
    parser.add_argument("--head_epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip_head_training", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output_dir")
    if not (0 < args.evidence_threshold < 1):
        parser.error("evidence_threshold must be in (0,1)")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    gap_model, cfg = load_gap_model(args.gap_model_path, device)
    gap_model.eval()
    gap_capture, hooks = make_feature_tap_gap(gap_model)
    dataset = RoadSkeletonDataset(args.root_path, split="val", image_size=None,
                                  source_patch_size=1500, return_full_image=True)
    # Import and construct the second (a400) model only after the 39e model has
    # been created; this preserves both implementations in one process.
    exp, probe_model, heads, probe_tap, probe_payload = load_probe_stack(
        Path(args.probe_repo), Path(args.probe_model_path), Path(args.probe_head_path), device)
    probe_model.eval(); heads.eval()
    args.output_dir.mkdir(parents=True)
    val_records, val_rows, val_totals = [], [], {}
    try:
        with torch.inference_mode():
            for record_index, tile in enumerate(tile_iter(dataset, "val", args.max_val_images, args.seed), 1):
                record = process_tile(tile, gap_model, probe_tap, heads, exp, gap_capture, cfg,
                                      args.candidate_pool, args.candidate_budget,
                                      args.evidence_threshold, device)
                rows, totals = proposal_rows(record, cfg, args.candidate_budget)
                val_rows.extend(rows)
                for name, counter in totals.items():
                    slot = val_totals.setdefault(name, Counter())
                    slot.update(counter)
                val_records.append(record)
                if record_index % 25 == 0:
                    print(f"[VAL-CANDIDATES] {record_index} tiles processed", flush=True)
        val_summary = {}
        for name, values in val_totals.items():
            val_summary[name] = dict(values)
            val_summary[name]["gap_recall"] = values["covered_gaps"] / values["verified_gaps"] if values["verified_gaps"] else None
            val_summary[name]["mean_candidates_per_tile"] = values["candidates"] / max(len(val_records), 1)
        write_csv(args.output_dir / "validation1_per_tile.csv", val_rows)
        if not args.skip_head_training:
            train_dataset = RoadSkeletonDataset(args.root_path, split="train", image_size=None,
                                                source_patch_size=1500, return_full_image=True)
            train_records = []
            with torch.inference_mode():
                for record_index, tile in enumerate(tile_iter(train_dataset, "train", args.max_train_images, args.seed), 1):
                    record = process_tile(tile, gap_model, probe_tap, heads, exp, gap_capture, cfg,
                                          args.candidate_pool, args.candidate_budget,
                                          args.evidence_threshold, device)
                    train_records.append(record)
                    if record_index % 25 == 0:
                        print(f"[HEAD-TRAIN-DATA] {record_index} crops extracted", flush=True)
            head_results, history = score_validation_heads(
                train_records, val_records, cfg, device, args.seed, args.head_epochs, args.candidate_budget)
            write_json(args.output_dir / "validation2_head_training.json", {
                "train_crops": len(train_records), "val_tiles": len(val_records),
                "input_features": {name: list(features) for name, features in SCORER_FEATURES.items()},
                "same_candidate_pool": True, "scorer_parameters": sum(p.numel() for p in CandidateScorer().parameters()),
                "epochs": args.head_epochs, "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
                "batch_size": 512, "positive_weight": "clip(negative_count/positive_count, 1, 20)",
                "history": history, "results": head_results,
                "scope": "Frozen backbones; candidate-level supervised diagnostic only. No surface writeback or retraining."})
        write_json(args.output_dir / "summary.json", {
            "gap_checkpoint": str(Path(args.gap_model_path).resolve()),
            "probe_backbone": str(Path(args.probe_model_path).resolve()),
            "probe_heads": str(Path(args.probe_head_path).resolve()),
            "probe_backbone_sha256": probe_payload.get("backbone_sha256"),
            "candidate_protocol": {"p2_threshold": cfg.threshold,
                                   "evidence_threshold": args.evidence_threshold,
                                   "gap_config": cfg.to_dict(),
                                   "candidate_pool": args.candidate_pool,
                                   "candidate_budget": args.candidate_budget,
                                   "gt_labeler": "existing losses.gap_losses.label_edges",
                                   "overlap_tiles": "512x512 stride256; tile-level counts include overlap duplicates"},
            "validation1": val_summary,
            "interpretation": "P2 is from 39e1398. E/H/F probe evidence is from the separate a4001b3 E2-direction checkpoint; this tests external evidence for candidate discovery, not an internal 39e encoder path or a causal direction-aggregation ablation.",
            "false_candidate_note": "no_local_path_negatives are the existing GT labeler's proxy for unsupported straight joins; they are not a dedicated parallel-road confusion classifier.",
        })
        print(json.dumps(val_summary, indent=2), flush=True)
    finally:
        for handle in hooks:
            handle.remove()
        probe_tap.close()
        del gap_model, probe_model, heads


if __name__ == "__main__":
    main()
