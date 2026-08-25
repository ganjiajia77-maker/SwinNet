import argparse
import csv
import os

import cv2
import numpy as np


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--baseline_pred_dir", type=str, required=True)
    parser.add_argument("--current_pred_dir", type=str, required=True)
    parser.add_argument("--baseline_name", type=str, default="baseline")
    parser.add_argument("--current_name", type=str, default="current")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--short_area_threshold", type=int, default=20)
    parser.add_argument("--output_csv", type=str, default="")
    return parser.parse_args()


def resolve_label_dir(root_path, split):
    for name in ("mask", "label"):
        path = os.path.join(root_path, split, name)
        if os.path.isdir(path):
            return path
    sibling = os.path.join(root_path, f"{split}_labels")
    if os.path.isdir(sibling):
        return sibling
    raise FileNotFoundError(f"Cannot find mask/label dir under {root_path}/{split}")


def list_prediction_files(pred_dir):
    surface_dir = os.path.join(pred_dir, "surface")
    if os.path.isdir(surface_dir):
        pred_dir = surface_dir
    files = [
        os.path.join(pred_dir, name)
        for name in sorted(os.listdir(pred_dir))
        if name.lower().endswith(IMAGE_EXTS)
    ]
    if not files:
        raise RuntimeError(f"No prediction images found in {pred_dir}")
    return files


def strip_prediction_suffix(path):
    base = os.path.splitext(os.path.basename(path))[0]
    suffixes = (
        "_surface_pred",
        "_mask_pred",
        "_pred",
        "_prediction",
        "_surface",
    )
    for suffix in suffixes:
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def find_label(label_dir, case_id):
    candidates = []
    bases = [
        case_id,
        case_id + "_sat",
        case_id + "_image",
        case_id + "_img",
        case_id.replace("_sat", ""),
        case_id.replace("_image", ""),
        case_id.replace("_img", ""),
    ]
    for base in dict.fromkeys(bases):
        for ext in IMAGE_EXTS:
            candidates.append(base + ext)
            candidates.append(base.replace("_sat", "_mask") + ext)
            candidates.append(base.replace("_image", "_mask") + ext)
            candidates.append(base.replace("_img", "_mask") + ext)
    for name in dict.fromkeys(candidates):
        path = os.path.join(label_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Cannot find GT label for prediction case: {case_id}")


def zhang_suen_skeletonize(mask_bool):
    image = mask_bool.astype(np.uint8).copy()
    while True:
        changed = False
        remove = []
        rows, cols = image.shape
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if image[y, x] == 0:
                    continue
                p2 = image[y - 1, x]
                p3 = image[y - 1, x + 1]
                p4 = image[y, x + 1]
                p5 = image[y + 1, x + 1]
                p6 = image[y + 1, x]
                p7 = image[y + 1, x - 1]
                p8 = image[y, x - 1]
                p9 = image[y - 1, x - 1]
                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                count = sum(neighbors)
                transitions = sum(
                    1
                    for a, b in zip(neighbors, neighbors[1:] + neighbors[:1])
                    if a == 0 and b == 1
                )
                if (
                    2 <= count <= 6
                    and transitions == 1
                    and p2 * p4 * p6 == 0
                    and p4 * p6 * p8 == 0
                ):
                    remove.append((y, x))
        if remove:
            changed = True
            for y, x in remove:
                image[y, x] = 0
        remove = []
        rows, cols = image.shape
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if image[y, x] == 0:
                    continue
                p2 = image[y - 1, x]
                p3 = image[y - 1, x + 1]
                p4 = image[y, x + 1]
                p5 = image[y + 1, x + 1]
                p6 = image[y + 1, x]
                p7 = image[y + 1, x - 1]
                p8 = image[y, x - 1]
                p9 = image[y - 1, x - 1]
                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                count = sum(neighbors)
                transitions = sum(
                    1
                    for a, b in zip(neighbors, neighbors[1:] + neighbors[:1])
                    if a == 0 and b == 1
                )
                if (
                    2 <= count <= 6
                    and transitions == 1
                    and p2 * p4 * p8 == 0
                    and p2 * p6 * p8 == 0
                ):
                    remove.append((y, x))
        if remove:
            changed = True
            for y, x in remove:
                image[y, x] = 0
        if not changed:
            break
    return image.astype(bool)


def skeletonize(mask_bool):
    if (
        hasattr(cv2, "ximgproc")
        and hasattr(cv2.ximgproc, "thinning")
        and hasattr(cv2.ximgproc, "THINNING_ZHANGSUEN")
    ):
        skel = cv2.ximgproc.thinning(
            (mask_bool.astype(np.uint8)) * 255,
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN,
        )
        return skel > 127
    return zhang_suen_skeletonize(mask_bool)


def component_stats(mask_bool, short_area_threshold):
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num > 1 else np.empty((0,), dtype=np.int32)
    total = float(areas.sum()) if areas.size else 0.0
    return {
        "components": float(num - 1),
        "short_components": float((areas < short_area_threshold).sum()) if areas.size else 0.0,
        "largest_ratio": float(areas.max() / total) if total > 0 else 0.0,
    }


def cldice(pred_bool, gt_bool):
    pred_skel = skeletonize(pred_bool)
    gt_skel = skeletonize(gt_bool)
    tprec = float((pred_skel & gt_bool).sum()) / (float(pred_skel.sum()) + 1e-8)
    tsens = float((gt_skel & pred_bool).sum()) / (float(gt_skel.sum()) + 1e-8)
    return (2.0 * tprec * tsens) / (tprec + tsens + 1e-8)


def evaluate_prediction_dir(name, pred_dir, label_dir, short_area_threshold):
    tp = fp = fn = 0.0
    cldice_values = []
    pred_comp = []
    gt_comp = []
    frag_idx = []
    extra_comp = []
    pred_short = []
    gt_short = []
    pred_largest = []
    gt_largest = []
    files = list_prediction_files(pred_dir)
    for pred_path in files:
        case_id = strip_prediction_suffix(pred_path)
        label_path = find_label(label_dir, case_id)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if pred is None:
            raise FileNotFoundError(pred_path)
        if gt is None:
            raise FileNotFoundError(label_path)
        pred_bool = pred > 127
        gt_bool = gt > 127
        if pred_bool.shape != gt_bool.shape:
            gt_bool = cv2.resize(
                gt_bool.astype(np.uint8),
                (pred_bool.shape[1], pred_bool.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        tp += float((pred_bool & gt_bool).sum())
        fp += float((pred_bool & (~gt_bool)).sum())
        fn += float(((~pred_bool) & gt_bool).sum())
        cldice_values.append(cldice(pred_bool, gt_bool))
        ps = component_stats(pred_bool, short_area_threshold)
        gs = component_stats(gt_bool, short_area_threshold)
        pred_comp.append(ps["components"])
        gt_comp.append(gs["components"])
        frag_idx.append(ps["components"] / max(gs["components"], 1.0))
        extra_comp.append(max(ps["components"] - gs["components"], 0.0))
        pred_short.append(ps["short_components"])
        gt_short.append(gs["short_components"])
        pred_largest.append(ps["largest_ratio"])
        gt_largest.append(gs["largest_ratio"])
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {
        "name": name,
        "pred_dir": pred_dir,
        "n_images": len(files),
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "cldice": float(np.mean(cldice_values)),
        "pred_comp": float(np.mean(pred_comp)),
        "gt_comp": float(np.mean(gt_comp)),
        "frag_idx": float(np.mean(frag_idx)),
        "extra_comp": float(np.mean(extra_comp)),
        "pred_short": float(np.mean(pred_short)),
        "gt_short": float(np.mean(gt_short)),
        "pred_largest_ratio": float(np.mean(pred_largest)),
        "gt_largest_ratio": float(np.mean(gt_largest)),
    }


def print_rows(rows):
    print("name | images | IoU | F1 | P | R | clDice | frag_idx↓ | extra_comp↓ | pred_comp | gt_comp | pred_short↓ | largest_ratio")
    for row in rows:
        print(
            f"{row['name']} | {row['n_images']} | {row['iou']:.4f} | {row['f1']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['cldice']:.4f} | "
            f"{row['frag_idx']:.3f} | {row['extra_comp']:.2f} | {row['pred_comp']:.2f} | "
            f"{row['gt_comp']:.2f} | {row['pred_short']:.2f} | {row['pred_largest_ratio']:.3f}"
        )
    if len(rows) == 2:
        base, cur = rows
        print("\nDelta current - baseline")
        for key in ("iou", "f1", "cldice", "frag_idx", "extra_comp", "pred_short", "pred_largest_ratio"):
            print(f"  {key}: {cur[key] - base[key]:+.4f}")


def main():
    args = parse_args()
    label_dir = resolve_label_dir(args.root_path, args.split)
    rows = [
        evaluate_prediction_dir(args.baseline_name, args.baseline_pred_dir, label_dir, args.short_area_threshold),
        evaluate_prediction_dir(args.current_name, args.current_pred_dir, label_dir, args.short_area_threshold),
    ]
    print_rows(rows)
    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
