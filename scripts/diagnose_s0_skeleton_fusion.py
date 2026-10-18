"""Inspect S_H and the direct F_H2 contribution on the archived S0 test cases.

The reference surface masks must match the recomputed masks exactly.  The
counterfactual removes only F_H2's additive contribution at the final
surface_feature_delta; the earlier B2/B3 interactions remain intact.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "road_comparison"))

from config import get_config
from evaluation.road_comparison.compare_predictions import case_id, index_files, read_binary
from evaluation.road_comparison.road_metrics import skeletonize
from networks.skeleton_feature_architecture import (
    require_checkpoint_architecture,
    restore_model_config,
)
from networks.vision_transformer import SwinUnet
from networks.swin_transformer_unet_skip_expand_decoder_sys import map_to_token


MODEL_NAME = "h0_skeleton_feature_sdgate_mass1500_random512_fp32_noema80_b2_acc1"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def index_images(directory):
    allowed = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in allowed:
            name = case_id(path)
            if name in images:
                raise ValueError(f"Duplicate test image case: {images[name]} and {path}")
            images[name] = path
    return images


def load_audit(audit, expected_cases):
    manifest = json.loads((audit / "comparison.json").read_text(encoding="utf-8"))
    selection = json.loads((audit / "val" / "val_selection.json").read_text(encoding="utf-8"))
    models = [item for item in manifest["models"] if item["name"] == MODEL_NAME]
    if len(models) != 1 or selection.get("selected_on") != "val":
        raise ValueError("Audit does not describe the S0 model and validation selection")
    model = models[0]
    for relative, expected_hash in model["provenance"].get("source_sha256", {}).items():
        source = ROOT / relative
        if not source.is_file() or sha256(source) != expected_hash:
            raise ValueError(f"Source code differs from the 49-case audit: {relative}")
    sources = model["predictions"]["test"]
    if len(sources) != 1:
        raise ValueError("Expected exactly one test prediction source")
    threshold = float(selection["models"][MODEL_NAME]["threshold"])
    if abs(threshold - float(sources[0]["threshold"])) > 1e-8:
        raise ValueError("Test mask threshold differs from the validation selection")
    pred_dir = Path(sources[0]["path"])
    if not pred_dir.is_absolute():
        pred_dir = audit / pred_dir
    pred = index_files(pred_dir)
    if len(pred) != expected_cases:
        raise ValueError(f"Expected {expected_cases} archived masks, found {len(pred)}")
    data_root = Path(manifest["data_root"])
    images = index_images(data_root / "test" / "image")
    labels_dir = data_root / "test" / "mask"
    if not labels_dir.is_dir():
        labels_dir = data_root / "test" / "label"
    labels = index_files(labels_dir)
    if set(pred) != set(images) or set(pred) != set(labels):
        raise ValueError("The 49 archived predictions, input images, and GT labels are not paired")
    checkpoint = Path(model["provenance"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Audit checkpoint is absent: {checkpoint}")
    return manifest, model, threshold, checkpoint, images, labels, pred


def build_model(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require_checkpoint_architecture(state)
    saved = argparse.Namespace(**state["args"])
    saved.cfg = str(ROOT / "configs" / "swin_tiny_patch4_window7_224_lite.yaml")
    config = restore_model_config(get_config(saved), state["model_config"])
    model = SwinUnet(
        config=config,
        img_size=saved.img_size,
        num_classes=1,
        return_skeleton=True,
        bottleneck_type=saved.bottleneck_type,
        final_topology_eta_init=saved.final_topology_eta_init,
        final_gap_rho_init=saved.final_gap_rho_init,
        structure_profile=saved.structure_profile,
        stage2_skeleton_gradient_ratio=saved.stage2_skeleton_gradient_ratio,
        stage3_skeleton_gradient_ratio=saved.stage3_skeleton_gradient_ratio,
        stage3_gate_topology_gradient_ratio=saved.stage3_gate_topology_gradient_ratio,
        final_skeleton_gradient_ratio=saved.final_skeleton_gradient_ratio,
        enable_highres_structure_stream=saved.enable_highres_structure_stream,
        highres_structure_channels=saved.highres_structure_channels,
        highres_structure_fuse_stages=saved.highres_structure_fuse_stages,
        highres_structure_fusion_mode=saved.highres_structure_fusion_mode,
        enable_post_refine_structure_interaction=saved.enable_post_refine_structure_interaction,
        enable_h3_surface_fusion=saved.enable_h3_surface_fusion,
        enable_global_topology=saved.enable_global_topology,
        global_topology_max_nodes=saved.global_topology_max_nodes,
        global_topology_heads=saved.global_topology_heads,
        global_topology_alpha_max=saved.global_topology_alpha_max,
        stage_skeleton_mode=saved.stage_skeleton_mode,
        remove_stage2_pre_topology_source=saved.remove_stage2_pre_topology_source,
    )
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, state.get("epoch"), saved


def positions(length, tile, stride):
    if length <= tile:
        return [0]
    return sorted(set(range(0, length - tile + 1, stride)) | {length - tile})


def mean_at(values, selected):
    return float(values[selected].mean()) if selected.any() else None


def rate_at(values, selected):
    return float(values[selected].mean()) if selected.any() else None


def infer_case(model, image_rgb, tile, stride, threshold, device):
    core = model.swin_unet
    h, w = image_rgb.shape[:2]
    if (h, w) != (1500, 1500) or tile != 512 or stride != 256:
        raise ValueError("This diagnosis requires the archived native1500 / tile512 / stride256 protocol")
    channels = ("surface", "no_h", "skeleton", "h_delta_norm", "h_proj_norm", "total_delta_norm")
    canvas = {key: torch.zeros((h, w), device=device) for key in channels}
    weight_canvas = torch.zeros((h, w), device=device)
    weight_1d = (1.0 - torch.linspace(-1.0, 1.0, steps=tile, device=device).abs()).clamp_min(0.05)
    tile_weight = torch.outer(weight_1d, weight_1d)
    captured = {}

    def capture_delta(_module, inputs, output):
        captured["concat"] = inputs[0]
        captured["delta"] = output

    def capture_h_proj(_module, _inputs, output):
        captured["h_proj"] = output

    hook_delta = core.surface_feature_delta.register_forward_hook(capture_delta)
    hook_h = core.surface_h_proj.register_forward_hook(capture_h_proj)
    try:
        with torch.inference_mode():
            for y in positions(h, tile, stride):
                for x in positions(w, tile, stride):
                    y2, x2 = min(y + tile, h), min(x + tile, w)
                    crop = image_rgb[y:y2, x:x2]
                    if crop.shape[:2] != (tile, tile):
                        crop = np.pad(crop, ((0, tile - crop.shape[0]),
                                             (0, tile - crop.shape[1]), (0, 0)))
                    inp = torch.from_numpy(((crop.astype(np.float32) / 255.0 - MEAN) / STD)
                                           .transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
                    captured.clear()
                    original = model(inp)[0]
                    sh = core.last_highres_structure_skeleton
                    if sh is None or set(captured) != {"concat", "delta", "h_proj"}:
                        raise RuntimeError("S_H or the final F_H2 fusion was not observed")
                    concat, delta, h_proj = captured["concat"], captured["delta"], captured["h_proj"]
                    decoder_channels = core.embed_dim
                    if concat.shape[1] != decoder_channels + 64 or h_proj.shape[1] != 32:
                        raise RuntimeError("Unexpected final fusion channel layout")
                    h_contrib = F.conv2d(h_proj, core.surface_feature_delta.weight[:, -32:])
                    decoder = concat[:, :decoder_channels]
                    modified = decoder + delta - h_contrib
                    tokens = map_to_token(modified)
                    expanded = core.up(tokens).view(1, tile, tile, -1).permute(0, 3, 1, 2)
                    no_h = core.guided_head(expanded, z_struct=core.last_stage_features["H2"])[0]
                    skeleton = F.interpolate(sh, size=(tile, tile), mode="bilinear", align_corners=False)
                    metrics = {
                        "surface": original[0, 0], "no_h": no_h[0, 0],
                        "skeleton": skeleton[0, 0],
                        "h_delta_norm": F.interpolate(h_contrib.square().mean(dim=1, keepdim=True).sqrt(),
                                                      size=(tile, tile), mode="bilinear", align_corners=False)[0, 0],
                        "h_proj_norm": F.interpolate(h_proj.square().mean(dim=1, keepdim=True).sqrt(),
                                                     size=(tile, tile), mode="bilinear", align_corners=False)[0, 0],
                        "total_delta_norm": F.interpolate(delta.square().mean(dim=1, keepdim=True).sqrt(),
                                                          size=(tile, tile), mode="bilinear", align_corners=False)[0, 0],
                    }
                    area_weight = tile_weight[:y2-y, :x2-x]
                    for key, value in metrics.items():
                        canvas[key][y:y2, x:x2] += value[:y2-y, :x2-x] * area_weight
                    weight_canvas[y:y2, x:x2] += area_weight
    finally:
        hook_delta.remove()
        hook_h.remove()
    if weight_canvas.min().item() <= 0:
        raise RuntimeError("A native image pixel was left uncovered")
    result = {key: (value / weight_canvas).cpu().numpy() for key, value in canvas.items()}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite diagnostic output")
    result["surface_prob"] = 1.0 / (1.0 + np.exp(-np.clip(result["surface"], -80, 80)))
    result["no_h_prob"] = 1.0 / (1.0 + np.exp(-np.clip(result["no_h"], -80, 80)))
    result["skeleton_prob"] = 1.0 / (1.0 + np.exp(-np.clip(result["skeleton"], -80, 80)))
    return result


def analyse_case(name, result, gt, archived, threshold, maps_dir):
    pred = result["surface_prob"] >= threshold
    if not np.array_equal(pred, archived):
        mismatch = int(np.count_nonzero(pred ^ archived))
        raise ValueError(f"{name}: recomputed mask differs from archived 49-case test ({mismatch} pixels)")
    no_h = result["no_h_prob"] >= threshold
    gt_center = skeletonize(gt)
    missed = gt_center & ~pred
    missed_gap3 = gt_center & ~cv2.dilate(pred.astype(np.uint8),
                                         np.ones((7, 7), np.uint8)).astype(bool)
    sh = result["skeleton_prob"]
    sh_positive = sh >= 0.5
    strong_missed = missed & sh_positive
    far_background = ~cv2.dilate(gt.astype(np.uint8),
                                  np.ones((11, 11), np.uint8)).astype(bool)
    logit_effect = result["surface"] - result["no_h"]
    helps = pred & ~no_h & gt_center
    harms = ~pred & no_h & gt_center
    row = {
        "case": name, "gt_center_pixels": int(gt_center.sum()),
        "missed_center_pixels": int(missed.sum()),
        "missed_gap3_pixels": int(missed_gap3.sum()),
        "sh_positive_on_missed": int(strong_missed.sum()),
        "sh_positive_on_gap3": int((missed_gap3 & sh_positive).sum()),
        "sh_recall_gt_center": rate_at(sh_positive, gt_center),
        "sh_recall_missed": rate_at(sh_positive, missed),
        "sh_recall_gap3": rate_at(sh_positive, missed_gap3),
        "sh_positive_far_background": rate_at(sh_positive, far_background),
        "sh_prob_missed_mean": mean_at(sh, missed),
        "sh_prob_detected_mean": mean_at(sh, gt_center & pred),
        "sh_prob_far_background_mean": mean_at(sh, far_background),
        "h_projection_norm_missed_mean": mean_at(result["h_proj_norm"], missed),
        "h_delta_norm_missed_mean": mean_at(result["h_delta_norm"], missed),
        "total_delta_norm_missed_mean": mean_at(result["total_delta_norm"], missed),
        "h_logit_effect_missed_mean": mean_at(logit_effect, missed),
        "h_logit_effect_strong_missed_mean": mean_at(logit_effect, strong_missed),
        "h_raises_logit_strong_missed": rate_at(logit_effect > 0, strong_missed),
        "h_helps_gt_center_pixels": int(helps.sum()),
        "h_harms_gt_center_pixels": int(harms.sum()),
        "h_helps_gt_road_pixels": int((pred & ~no_h & gt).sum()),
        "h_harms_gt_road_pixels": int((~pred & no_h & gt).sum()),
        "surface_tp": int((pred & gt).sum()),
        "surface_fp": int((pred & ~gt).sum()),
        "surface_fn": int((~pred & gt).sum()),
        "no_h_tp": int((no_h & gt).sum()),
        "no_h_fp": int((no_h & ~gt).sum()),
        "no_h_fn": int((~no_h & gt).sum()),
    }
    maps_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(maps_dir / f"{name}_sh_prob.png"), np.clip(np.round(sh * 255), 0, 255).astype(np.uint8))
    # Red means the F_H2 contribution raises the road logit; blue means it lowers it.
    clipped = np.clip(logit_effect, -1.0, 1.0)
    color = np.zeros((*clipped.shape, 3), dtype=np.uint8)
    color[:, :, 2] = np.round(np.clip(clipped, 0, 1) * 255).astype(np.uint8)
    color[:, :, 0] = np.round(np.clip(-clipped, 0, 1) * 255).astype(np.uint8)
    color[missed, 1] = 120
    color[strong_missed, 1] = 255
    cv2.imwrite(str(maps_dir / f"{name}_h_effect_missed.png"), color)
    return row


def aggregate(rows, threshold):
    def total(key):
        return sum(row[key] for row in rows)
    def mean(key):
        values = [row[key] for row in rows if row[key] is not None]
        return float(np.mean(values)) if values else None
    missed = total("missed_center_pixels")
    gap3 = total("missed_gap3_pixels")
    base_tp, base_fp, base_fn = (total(f"surface_{key}") for key in ("tp", "fp", "fn"))
    no_h_tp, no_h_fp, no_h_fn = (total(f"no_h_{key}") for key in ("tp", "fp", "fn"))
    return {
        "cases": len(rows), "surface_threshold_from_validation": threshold,
        "sh_threshold": 0.5, "missed_center_pixels": missed,
        "missed_gap3_pixels": gap3,
        "sh_positive_missed_fraction": total("sh_positive_on_missed") / missed if missed else None,
        "sh_positive_gap3_fraction": total("sh_positive_on_gap3") / gap3 if gap3 else None,
        "sh_prob_missed_mean_per_image": mean("sh_prob_missed_mean"),
        "sh_prob_detected_mean_per_image": mean("sh_prob_detected_mean"),
        "sh_prob_far_background_mean_per_image": mean("sh_prob_far_background_mean"),
        "h_projection_norm_missed_mean_per_image": mean("h_projection_norm_missed_mean"),
        "h_delta_norm_missed_mean_per_image": mean("h_delta_norm_missed_mean"),
        "total_delta_norm_missed_mean_per_image": mean("total_delta_norm_missed_mean"),
        "h_logit_effect_strong_missed_mean_per_image": mean("h_logit_effect_strong_missed_mean"),
        "h_raises_logit_strong_missed_mean_per_image": mean("h_raises_logit_strong_missed"),
        "h_helps_gt_center_pixels": total("h_helps_gt_center_pixels"),
        "h_harms_gt_center_pixels": total("h_harms_gt_center_pixels"),
        "h_helps_gt_road_pixels": total("h_helps_gt_road_pixels"),
        "h_harms_gt_road_pixels": total("h_harms_gt_road_pixels"),
        "surface_iou": base_tp / (base_tp + base_fp + base_fn),
        "no_h_iou": no_h_tp / (no_h_tp + no_h_fp + no_h_fn),
        "counterfactual_scope": "remove only the direct F_H2 channels from the final surface_feature_delta; earlier F_H2/B2/B3 interactions remain",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True, help="Existing 49-case unified evaluation directory")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_cases", type=int, default=49)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest, audit_model, threshold, checkpoint, images, labels, archived = load_audit(
        args.audit, args.expected_cases)
    device = torch.device(args.device)
    model, epoch, saved = build_model(checkpoint, device)
    if epoch != audit_model["provenance"]["checkpoint_epoch"]:
        raise ValueError("Checkpoint epoch differs from the 49-case audit")
    if saved.img_size != 512 or saved.source_patch_size != 1500:
        raise ValueError("Checkpoint input protocol differs from the 49-case audit")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, name in enumerate(sorted(archived), 1):
        image_bgr = cv2.imread(str(images[name]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise OSError(images[name])
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gt = read_binary(labels[name], 1500, allow_replicated_rgb=True)
        reference = read_binary(archived[name], 1500)
        result = infer_case(model, image, 512, 256, threshold, device)
        row = analyse_case(name, result, gt, reference, threshold, args.output_dir / "maps")
        rows.append(row)
        print(f"[{index}/{len(archived)}] {name}: missed={row['missed_center_pixels']}, "
              f"S_H-positive={row['sh_positive_on_missed']}, "
              f"H2-logit-effect={row['h_logit_effect_strong_missed_mean']}", flush=True)
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate(rows, threshold)
    summary.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                   checkpoint_epoch=epoch, audit=str(args.audit),
                   archived_masks_exact_match=True,
                   protocol="native1500; 512 overlap stride256; weighted tile logits; no TTA")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
