"""Diagnose S_H/S3 disagreement against verified local road breaks on S0 val.

This is an analysis of a frozen checkpoint. It uses the original validation
threshold and reproduces every archived validation surface mask exactly.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "road_comparison"))

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from evaluation.road_comparison.compare_predictions import index_files, read_binary
from evaluation.road_comparison.road_metrics import skeletonize
from scripts.diagnose_s0_skeleton_fusion import (
    MEAN, MODEL_NAME, STD, build_model, index_images, positions, sha256,
)


def load_val_audit(audit, expected_test_cases):
    manifest = json.loads((audit / "comparison.json").read_text(encoding="utf-8"))
    selection = json.loads((audit / "val" / "val_selection.json").read_text(encoding="utf-8"))
    models = [entry for entry in manifest["models"] if entry["name"] == MODEL_NAME]
    if len(models) != 1 or selection.get("selected_on") != "val":
        raise ValueError("Expected the S0 validation selection and manifest")
    model = models[0]
    for relative, expected in model["provenance"].get("source_sha256", {}).items():
        source = ROOT / relative
        if not source.is_file() or sha256(source) != expected:
            raise ValueError(f"Code differs from the archived S0 evaluation: {relative}")
    checkpoint = Path(model["provenance"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    test = model["predictions"].get("test", [])
    if len(test) != 1 or len(index_files(Path(test[0]["path"]))) != expected_test_cases:
        raise ValueError(f"This audit does not contain the requested {expected_test_cases}-case test result")
    threshold = float(selection["models"][MODEL_NAME]["threshold"])
    candidates = [entry for entry in model["predictions"]["val"]
                  if abs(float(entry["threshold"]) - threshold) < 1e-8]
    if len(candidates) != 1:
        raise ValueError("Selected validation threshold mask is absent or ambiguous")
    archived = index_files(Path(candidates[0]["path"]))
    data_root = Path(manifest["data_root"])
    image_dir = RoadSkeletonDataset._resolve_image_dir(str(data_root), "val")
    label_dir = RoadSkeletonDataset._resolve_label_dir(str(data_root), "val", ("mask", "label"))
    images, labels = index_images(Path(image_dir)), index_files(Path(label_dir))
    if set(images) != set(labels) or set(images) != set(archived):
        raise ValueError("Validation images, GT masks, and saved predictions are not paired")
    if selection.get("val_images") != len(images):
        raise ValueError("Validation case count differs from the threshold selection")
    return threshold, checkpoint, model, images, labels, archived


def infer_aligned(model, rgb, device):
    height, width = rgb.shape[:2]
    if (height, width) != (1500, 1500):
        raise ValueError("Expected a native 1500x1500 Massachusetts validation image")
    tile, stride = 512, 256
    weights_1d = (1 - torch.linspace(-1, 1, steps=tile, device=device).abs()).clamp_min(0.05)
    tile_weight = torch.outer(weights_1d, weights_1d)
    fields = {name: torch.zeros((height, width), device=device)
              for name in ("surface", "sh", "s3")}
    weights = torch.zeros((height, width), device=device)
    with torch.inference_mode():
        for y in positions(height, tile, stride):
            for x in positions(width, tile, stride):
                crop = rgb[y:y+tile, x:x+tile]
                inp = torch.from_numpy(((crop.astype(np.float32) / 255.0 - MEAN) / STD)
                                       .transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
                outputs = model(inp)
                if not isinstance(outputs, tuple) or not isinstance(outputs[-1], list):
                    raise RuntimeError("S0 did not return stage structure outputs")
                stage3 = [entry for entry in outputs[-1]
                          if entry.get("stage") == 3 and entry.get("refinement_step") == 1]
                if len(stage3) != 1 or stage3[0].get("skeleton") is None:
                    raise RuntimeError("Could not identify the post-refinement S3 logit")
                sh = model.swin_unet.last_highres_structure_skeleton
                if sh is None:
                    raise RuntimeError("Could not identify the final F_H2-derived S_H logit")
                logits = {
                    "surface": outputs[0][0, 0],
                    "sh": torch.nn.functional.interpolate(sh, size=(tile, tile),
                                                              mode="bilinear", align_corners=False)[0, 0],
                    "s3": torch.nn.functional.interpolate(stage3[0]["skeleton"],
                                                              size=(tile, tile), mode="bilinear",
                                                              align_corners=False)[0, 0],
                }
                for name, value in logits.items():
                    fields[name][y:y+tile, x:x+tile] += value * tile_weight
                weights[y:y+tile, x:x+tile] += tile_weight
    if weights.min().item() <= 0:
        raise RuntimeError("An image pixel is uncovered by the original sliding tiles")
    return {name: torch.sigmoid(field / weights).cpu().numpy()
            for name, field in fields.items()}


def local_gap_components(gt_center, surface, tolerance, margin, min_pixels):
    kernel = np.ones((2*tolerance+1, 2*tolerance+1), np.uint8)
    covered_surface = cv2.dilate(surface.astype(np.uint8), kernel).astype(bool)
    covered_center = gt_center & covered_surface
    missing = gt_center & ~covered_surface
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        missing.astype(np.uint8), connectivity=8)
    verified = np.zeros_like(missing)
    records = []
    for component_id in range(1, count):
        left, top, width, height, area = [int(v) for v in stats[component_id]]
        if area < min_pixels:
            records.append(dict(component_id=component_id, x=left, y=top, width=width,
                                height=height, pixels=area, anchor_groups=0,
                                status="too_short"))
            continue
        y0, y1 = max(0, top-margin), min(missing.shape[0], top+height+margin)
        x0, x1 = max(0, left-margin), min(missing.shape[1], left+width+margin)
        local_missing = components[y0:y1, x0:x1] == component_id
        local_pred = surface[y0:y1, x0:x1]
        boundary = (cv2.dilate(local_missing.astype(np.uint8),
                               np.ones((3, 3), np.uint8)).astype(bool)
                    & covered_center[y0:y1, x0:x1] & ~local_missing)
        anchor_count, anchor_labels, _, anchor_centers = (
            cv2.connectedComponentsWithStats(boundary.astype(np.uint8), connectivity=8))
        _, pred_labels = cv2.connectedComponents(local_pred.astype(np.uint8), connectivity=8)
        anchors = []
        for anchor_id in range(1, anchor_count):
            anchor = anchor_labels == anchor_id
            nearby_pred = cv2.dilate(anchor.astype(np.uint8), kernel).astype(bool) & local_pred
            road_components = set(int(v) for v in np.unique(pred_labels[nearby_pred]) if v > 0)
            if road_components:
                anchors.append((anchor_centers[anchor_id], road_components))
        if len(anchors) < 2:
            status = "one_sided_or_no_anchor"
        else:
            separated = False
            broken = False
            for i, (first, first_ids) in enumerate(anchors):
                for second, second_ids in anchors[i+1:]:
                    if float(np.linalg.norm(first-second)) < 4.0:
                        continue
                    separated = True
                    if first_ids.isdisjoint(second_ids):
                        broken = True
            status = ("verified_break" if broken else
                      "locally_connected" if separated else "anchors_too_close")
        if status == "verified_break":
            verified[y0:y1, x0:x1] |= local_missing
        records.append(dict(component_id=component_id, x=left, y=top, width=width,
                            height=height, pixels=area, anchor_groups=len(anchors),
                            status=status))
    return verified, missing, components, records


def save_gap_examples(name, rgb, gt_center, surface, sh, s3, component_labels,
                      gaps, maps_dir):
    touched = sorted((g for g in gaps if g["status"] == "verified_break"
                      and g.get("difference_detected")),
                     key=lambda g: g["pixels"], reverse=True)[:3]
    missed = sorted((g for g in gaps if g["status"] == "verified_break"
                     and g.get("jointly_missed")),
                    key=lambda g: g["pixels"], reverse=True)[:2]
    examples = {g["component_id"]: g for g in touched + missed}
    directory = maps_dir / "gap_examples"
    directory.mkdir(parents=True, exist_ok=True)
    for gap in examples.values():
        size = 256
        center_x = gap["x"] + gap["width"]//2
        center_y = gap["y"] + gap["height"]//2
        x0 = max(0, min(rgb.shape[1]-size, center_x-size//2))
        y0 = max(0, min(rgb.shape[0]-size, center_y-size//2))
        sl = np.s_[y0:y0+size, x0:x0+size]
        current = component_labels[sl] == gap["component_id"]
        line = gt_center[sl]
        image = cv2.cvtColor(rgb[sl], cv2.COLOR_RGB2BGR).copy()
        image[line] = (255, 255, 0)
        image[current] = (0, 0, 255)
        surface_panel = cv2.cvtColor(surface[sl].astype(np.uint8)*255,
                                     cv2.COLOR_GRAY2BGR)
        surface_panel[line] = (255, 255, 0)
        surface_panel[current] = (0, 0, 255)
        sh_panel = cv2.applyColorMap(np.round(sh[sl]*255).astype(np.uint8),
                                     cv2.COLORMAP_TURBO)
        s3_panel = cv2.applyColorMap(np.round(s3[sl]*255).astype(np.uint8),
                                     cv2.COLORMAP_TURBO)
        sh_panel[current] = (255, 255, 255)
        s3_panel[current] = (255, 255, 255)
        panels = [image, surface_panel, sh_panel, s3_panel]
        for panel, label in zip(panels, ("IMAGE + GT", "SURFACE + GT", "S_H", "S3")):
            cv2.putText(panel, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(panel, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (255, 255, 255), 1, cv2.LINE_AA)
        path = directory / f"{name}_gap_{gap['component_id']}.png"
        if not cv2.imwrite(str(path), np.concatenate(panels, axis=1)):
            raise OSError(path)
        gap["example_png"] = str(path)


def measure_case(name, rgb, aligned, gt, archived, threshold, tolerance, margin, min_pixels,
                 maps_dir):
    surface = aligned["surface"] >= threshold
    if not np.array_equal(surface, archived):
        raise ValueError(f"{name}: surface replay differs from archived validation mask by "
                         f"{int(np.count_nonzero(surface ^ archived))} pixels")
    sh, s3 = aligned["sh"], aligned["s3"]
    sh_binary, s3_binary = sh >= 0.5, s3 >= 0.5
    disagreement = sh_binary ^ s3_binary
    proposal = disagreement & ~surface
    gt_center = skeletonize(gt)
    verified, missing, missing_components, gaps = local_gap_components(
        gt_center, surface, tolerance, margin, min_pixels)
    distance, nearest = ndimage.distance_transform_edt(~gt_center, return_indices=True)
    near_center = distance <= tolerance
    nearest_gap = verified[nearest[0], nearest[1]]
    nearest_covered = (~missing & gt_center)[nearest[0], nearest[1]]
    labels = {
        "verified_break": proposal & near_center & nearest_gap,
        "covered_or_shifted": proposal & near_center & ~nearest_gap & nearest_covered,
        "unverified_centerline_miss": proposal & near_center & ~nearest_gap & ~nearest_covered,
        "gt_road_off_centerline": proposal & ~near_center & gt,
        "background": proposal & ~near_center & ~gt,
    }
    if sum(int(mask.sum()) for mask in labels.values()) != int(proposal.sum()):
        raise AssertionError("Proposal categories do not partition the disagreement")
    proposal_components, proposal_component_labels = cv2.connectedComponents(
        proposal.astype(np.uint8), connectivity=8)
    component_sizes = np.bincount(proposal_component_labels[proposal],
                                  minlength=proposal_components)
    category_sizes = {key: np.bincount(proposal_component_labels[mask],
                                      minlength=proposal_components)
                      for key, mask in labels.items()}
    component_metrics = {
        "proposal_components": proposal_components-1,
        "components_touch_verified_break": int(np.count_nonzero(
            category_sizes["verified_break"][1:])),
        "components_only_background": int(np.count_nonzero(
            (category_sizes["background"][1:] == component_sizes[1:])
            & (component_sizes[1:] > 0))),
        "components_only_covered_or_shifted": int(np.count_nonzero(
            (category_sizes["covered_or_shifted"][1:] == component_sizes[1:])
            & (component_sizes[1:] > 0))),
    }
    tol_kernel = np.ones((2*tolerance+1, 2*tolerance+1), np.uint8)
    covered_gt_center = gt_center & cv2.dilate(
        surface.astype(np.uint8), tol_kernel).astype(bool)
    missing_distance, missing_nearest = ndimage.distance_transform_edt(
        ~missing, return_indices=True)
    nearest_missing_id = missing_components[missing_nearest[0], missing_nearest[1]]
    if sum(g["pixels"] for g in gaps if g["status"] == "verified_break") != int(verified.sum()):
        raise AssertionError("Verified gap components must partition the verified pixel map")
    verified_count = 0
    for gap in gaps:
        if gap["status"] != "verified_break":
            continue
        verified_count += 1
        x, y, w, h = gap["x"], gap["y"], gap["width"], gap["height"]
        x0, x1 = max(0, x-tolerance), min(gt.shape[1], x+w+tolerance)
        y0, y1 = max(0, y-tolerance), min(gt.shape[0], y+h+tolerance)
        region = missing_components[y0:y1, x0:x1] == gap["component_id"]
        if int(region.sum()) != gap["pixels"]:
            raise AssertionError("A verified gap must contain only its own component pixels")
        owned = ((nearest_missing_id[y0:y1, x0:x1] == gap["component_id"])
                 & (missing_distance[y0:y1, x0:x1] <= tolerance))
        def aligned_response(binary):
            source = binary[y0:y1, x0:x1] & owned
            return region & cv2.dilate(source.astype(np.uint8), tol_kernel).astype(bool)
        active = aligned_response(proposal)
        sh_active = aligned_response(sh_binary)
        s3_active = aligned_response(s3_binary)
        boundary = (cv2.dilate(region.astype(np.uint8),
                               np.ones((3, 3), np.uint8)).astype(bool)
                    & covered_gt_center[y0:y1, x0:x1])
        if not boundary.any():
            raise AssertionError("A verified break must have GT centerline on both sides")
        edge = region & (ndimage.distance_transform_edt(~boundary) <= 4.0)
        interior = region & ~edge
        own_sh = sh[y0:y1, x0:x1][region]
        own_s3 = s3[y0:y1, x0:x1][region]
        gap.update(case=name, difference_detected=bool(active.any()),
                   jointly_missed=not sh_active.any() and not s3_active.any(),
                   sh_detected=bool(sh_active.any()), s3_detected=bool(s3_active.any()),
                   difference_covered_pixels=int(active.sum()),
                   jointly_missed_pixels=int((region & ~sh_active & ~s3_active).sum()),
                   edge_pixels=int(edge.sum()), interior_pixels=int(interior.sum()),
                   difference_edge_pixels=int((active & edge).sum()),
                   difference_interior_pixels=int((active & interior).sum()),
                   sh_edge_pixels=int((sh_active & edge).sum()),
                   sh_interior_pixels=int((sh_active & interior).sum()),
                   s3_edge_pixels=int((s3_active & edge).sum()),
                   s3_interior_pixels=int((s3_active & interior).sum()),
                   sh_direct_centerline_pixels=int((region & sh_binary[y0:y1, x0:x1]).sum()),
                   sh_nearby_only_pixels=int((sh_active & ~sh_binary[y0:y1, x0:x1]).sum()),
                   s3_direct_centerline_pixels=int((region & s3_binary[y0:y1, x0:x1]).sum()),
                   s3_nearby_only_pixels=int((s3_active & ~s3_binary[y0:y1, x0:x1]).sum()),
                   sh_prob_mean=float(own_sh.mean()),
                   s3_prob_mean=float(own_s3.mean()))
    row = dict(case=name, proposals=int(proposal.sum()),
               **component_metrics,
               sh_only_proposals=int((sh_binary & ~s3_binary & ~surface).sum()),
               s3_only_proposals=int((s3_binary & ~sh_binary & ~surface).sum()),
               **{f"proposal_{key}": int(value.sum()) for key, value in labels.items()},
               gt_centerline_pixels=int(gt_center.sum()),
               missing_centerline_pixels=int(missing.sum()),
               verified_break_pixels=int(verified.sum()),
               verified_breaks=verified_count,
               difference_detected_breaks=sum(g.get("difference_detected", False) for g in gaps),
               jointly_missed_breaks=sum(g.get("jointly_missed", False) for g in gaps),
               sh_detected_breaks=sum(g.get("sh_detected", False) for g in gaps),
               s3_detected_breaks=sum(g.get("s3_detected", False) for g in gaps),
               both_detected_breaks=sum(g.get("sh_detected", False) and
                                        g.get("s3_detected", False) for g in gaps),
               difference_covered_break_pixels=sum(g.get("difference_covered_pixels", 0) for g in gaps),
               jointly_missed_break_pixels=sum(g.get("jointly_missed_pixels", 0) for g in gaps),
               unverified_missing_components=sum(g["status"] != "verified_break" for g in gaps))
    maps_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(maps_dir / f"{name}_sh_prob.png"), np.round(sh*255).astype(np.uint8))
    cv2.imwrite(str(maps_dir / f"{name}_s3_prob.png"), np.round(s3*255).astype(np.uint8))
    visual = np.zeros((*gt.shape, 3), np.uint8)
    visual[missing] = (80, 80, 80)
    visual[verified] = (255, 255, 255)
    colors = dict(verified_break=(0, 0, 255), covered_or_shifted=(255, 0, 0),
                  unverified_centerline_miss=(0, 255, 255),
                  gt_road_off_centerline=(255, 255, 0), background=(255, 0, 255))
    for key, mask in labels.items():
        visual[mask] = colors[key]
    cv2.imwrite(str(maps_dir / f"{name}_proposal_classes.png"), visual)
    save_gap_examples(name, rgb, gt_center, surface, sh, s3,
                      missing_components, gaps, maps_dir)
    return row, gaps


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, gaps, threshold, tolerance, margin, checkpoint, epoch, audit):
    def total(key):
        return sum(int(row.get(key, 0)) for row in rows)
    def gap_total(key):
        return sum(int(gap.get(key, 0)) for gap in gaps)
    proposals, breaks = total("proposals"), total("verified_breaks")
    proposal_components = total("proposal_components")
    verified_pixels = total("verified_break_pixels")
    edge_pixels, interior_pixels = gap_total("edge_pixels"), gap_total("interior_pixels")
    return dict(cases=len(rows), checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                checkpoint_epoch=epoch, audit=str(audit), surface_threshold=threshold,
                skeleton_threshold=0.5, centerline_tolerance_px=tolerance,
                local_continuation_margin_px=margin,
                proposal_definition="(S_H>=0.5 XOR S3>=0.5) AND surface<threshold",
                true_break_definition="GT centerline uncovered within tolerance; two separated covered GT-centerline anchor groups; locally disconnected predicted road components",
                archived_val_masks_exact_match=True,
                proposals=proposals,
                proposal_components=proposal_components,
                components_touch_verified_break=total("components_touch_verified_break"),
                components_touch_verified_break_fraction=(
                    total("components_touch_verified_break")/proposal_components
                    if proposal_components else None),
                components_only_background=total("components_only_background"),
                components_only_covered_or_shifted=total("components_only_covered_or_shifted"),
                proposal_verified_break=total("proposal_verified_break"),
                proposal_verified_break_fraction=total("proposal_verified_break")/proposals if proposals else None,
                proposal_covered_or_shifted=total("proposal_covered_or_shifted"),
                proposal_unverified_centerline_miss=total("proposal_unverified_centerline_miss"),
                proposal_gt_road_off_centerline=total("proposal_gt_road_off_centerline"),
                proposal_background=total("proposal_background"),
                verified_breaks=breaks,
                difference_detected_breaks=total("difference_detected_breaks"),
                difference_detected_break_fraction=total("difference_detected_breaks")/breaks if breaks else None,
                jointly_missed_breaks=total("jointly_missed_breaks"),
                jointly_missed_break_fraction=total("jointly_missed_breaks")/breaks if breaks else None,
                both_detected_breaks=total("both_detected_breaks"),
                verified_break_pixels=verified_pixels,
                difference_covered_break_pixels=total("difference_covered_break_pixels"),
                difference_covered_break_pixel_fraction=(
                    total("difference_covered_break_pixels")/verified_pixels
                    if verified_pixels else None),
                jointly_missed_break_pixels=total("jointly_missed_break_pixels"),
                jointly_missed_break_pixel_fraction=(
                    total("jointly_missed_break_pixels")/verified_pixels
                    if verified_pixels else None),
                edge_break_pixels=edge_pixels, interior_break_pixels=interior_pixels,
                sh_edge_coverage=(gap_total("sh_edge_pixels")/edge_pixels if edge_pixels else None),
                sh_interior_coverage=(gap_total("sh_interior_pixels")/interior_pixels
                                      if interior_pixels else None),
                s3_edge_coverage=(gap_total("s3_edge_pixels")/edge_pixels if edge_pixels else None),
                s3_interior_coverage=(gap_total("s3_interior_pixels")/interior_pixels
                                      if interior_pixels else None),
                difference_edge_coverage=(gap_total("difference_edge_pixels")/edge_pixels
                                          if edge_pixels else None),
                difference_interior_coverage=(gap_total("difference_interior_pixels")/interior_pixels
                                              if interior_pixels else None),
                sh_direct_centerline_pixels=gap_total("sh_direct_centerline_pixels"),
                sh_nearby_only_pixels=gap_total("sh_nearby_only_pixels"),
                s3_direct_centerline_pixels=gap_total("s3_direct_centerline_pixels"),
                s3_nearby_only_pixels=gap_total("s3_nearby_only_pixels"),
                unverified_missing_components=total("unverified_missing_components"),
                gap_status_counts={status: sum(g["status"] == status for g in gaps)
                                   for status in sorted({g["status"] for g in gaps})})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--centerline_tolerance", type=int, default=3)
    parser.add_argument("--local_margin", type=int, default=32)
    parser.add_argument("--min_missing_pixels", type=int, default=2)
    parser.add_argument("--expected_test_cases", type=int, default=49)
    args = parser.parse_args()
    if args.centerline_tolerance < 1 or args.local_margin < args.centerline_tolerance:
        parser.error("Require positive centerline tolerance and a larger local margin")
    threshold, checkpoint, audit_model, images, labels, archived = load_val_audit(
        args.audit, args.expected_test_cases)
    model, epoch, saved = build_model(checkpoint, torch.device(args.device))
    if epoch != audit_model["provenance"]["checkpoint_epoch"]:
        raise ValueError("Checkpoint epoch differs from archived 49-case evaluation")
    if saved.img_size != 512 or saved.source_patch_size != 1500:
        raise ValueError("Checkpoint input size is incompatible with the archived evaluation")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, gaps = [], []
    for index, name in enumerate(sorted(images), 1):
        bgr = cv2.imread(str(images[name]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(images[name])
        gt = read_binary(labels[name], 1500, allow_replicated_rgb=True)
        reference = read_binary(archived[name], 1500)
        aligned = infer_aligned(model, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                torch.device(args.device))
        row, image_gaps = measure_case(name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                       aligned, gt, reference, threshold,
                                       args.centerline_tolerance, args.local_margin,
                                       args.min_missing_pixels, args.output_dir / "maps")
        rows.append(row)
        gaps.extend(image_gaps)
        print(f"[{index}/{len(images)}] {name}: proposals={row['proposals']}, "
              f"verified_breaks={row['verified_breaks']}, "
              f"difference_detected={row['difference_detected_breaks']}, "
              f"jointly_missed={row['jointly_missed_breaks']}", flush=True)
    write_csv(args.output_dir / "per_image.csv", rows)
    write_csv(args.output_dir / "gap_components.csv", gaps)
    report = aggregate(rows, gaps, threshold, args.centerline_tolerance,
                       args.local_margin, checkpoint, epoch, args.audit)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
