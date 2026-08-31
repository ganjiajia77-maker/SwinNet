import argparse
import contextlib
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_structure_supervision import load_model
from datasets.dataset_road_skeleton import RoadSkeletonDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare final surface logits with boundary correction ON versus "
            "OFF by temporarily setting guided_head.beta to zero."
        )
    )
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--cfg", type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache_mode", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--accumulation_steps", type=int, default=0)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--amp_opt_level", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--dataset", type=str, default="ImageData")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--structure_profile", type=str, default="stage23_boundary_0626")
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
    parser.add_argument("--final_topology_eta_init", type=float, default=0.0)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.0)
    parser.add_argument("--stage_topology_stages", type=str, default="none")
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    parser.add_argument("--stage_topology_bias_mode", type=str, default="pairwise_skeleton")
    parser.add_argument("--stage_topology_ratio", type=float, default=0.08)
    parser.add_argument("--stage_topology_topo_clip", type=float, default=4.0)
    parser.add_argument("--stage2_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--stage3_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--final_skeleton_gradient_ratio", type=float, default=0.0)
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23")
    parser.add_argument("--highres_structure_fusion_mode", type=str, default="stage23")
    parser.add_argument("--model_impl", type=str, default="auto", choices=("auto", "standard", "selective"))
    return parser.parse_args()


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_guided_head(model):
    module = model.module if hasattr(model, "module") else model
    swin = getattr(module, "swin_unet", module)
    guided_head = getattr(swin, "guided_head", None)
    if guided_head is None:
        raise RuntimeError("Could not find swin_unet.guided_head")
    return guided_head


@contextlib.contextmanager
def boundary_beta_zero(model):
    guided_head = get_guided_head(model)
    if not hasattr(guided_head, "beta"):
        raise RuntimeError("guided_head has no beta parameter for boundary residual")
    original = guided_head.beta.detach().clone()
    with torch.no_grad():
        guided_head.beta.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            guided_head.beta.copy_(original)


def resize_like(tensor, reference, mode="nearest"):
    if tensor.shape[-2:] == reference.shape[-2:]:
        return tensor
    kwargs = {"size": reference.shape[-2:], "mode": mode}
    if mode in ("bilinear", "bicubic", "trilinear"):
        kwargs["align_corners"] = False
    return F.interpolate(tensor.float(), **kwargs)


def classification_metrics(pred, gt):
    tp = int((pred & gt).sum().item())
    fp = int((pred & (~gt)).sum().item())
    fn = int(((~pred) & gt).sum().item())
    tn = int(((~pred) & (~gt)).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
    }


def empty_counts():
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def add_counts(total, metrics):
    for key in total:
        total[key] += metrics[key]


def summarize_counts(total):
    precision = total["tp"] / max(total["tp"] + total["fp"], 1)
    recall = total["tp"] / max(total["tp"] + total["fn"], 1)
    iou = total["tp"] / max(total["tp"] + total["fp"] + total["fn"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, iou, f1


def main():
    args = parse_args()
    set_deterministic(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    model.eval()
    guided_head = get_guided_head(model)
    beta_value = float(guided_head.beta.detach().cpu()) if hasattr(guided_head, "beta") else float("nan")

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and args.num_workers > 0,
    )

    off_total = empty_counts()
    on_total = empty_counts()
    transitions = {
        "FN_to_TP": 0,
        "FP_to_TN": 0,
        "TN_to_FP": 0,
        "TP_to_FN": 0,
        "changed": 0,
    }
    delta_abs_values = []
    delta_values = []
    off_std_values = []
    batches = 0
    images = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Boundary delta quadrants")):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            batches += 1
            images += int(batch["image"].shape[0])
            image = batch["image"].to(device)
            mask = (batch["mask"].to(device) > 0.5).float()
            skeleton = batch["skeleton"].to(device).float()

            with boundary_beta_zero(model):
                off_logits = model(
                    image,
                    gt_skeleton=skeleton,
                    topology_alpha_scale=1.0,
                    teacher_forcing_ratio=0.0,
                )[0]
            on_logits = model(
                image,
                gt_skeleton=skeleton,
                topology_alpha_scale=1.0,
                teacher_forcing_ratio=0.0,
            )[0]

            gt = resize_like(mask, off_logits, mode="nearest") > 0.5
            on_logits = resize_like(on_logits, off_logits, mode="bilinear")
            delta = on_logits - off_logits
            off_pred = torch.sigmoid(off_logits) >= args.threshold
            on_pred = torch.sigmoid(on_logits) >= args.threshold

            add_counts(off_total, classification_metrics(off_pred, gt))
            add_counts(on_total, classification_metrics(on_pred, gt))

            fn_off = gt & (~off_pred)
            fp_off = (~gt) & off_pred
            tp_off = gt & off_pred
            tn_off = (~gt) & (~off_pred)
            changed = off_pred != on_pred

            transitions["changed"] += int(changed.sum().item())
            transitions["FN_to_TP"] += int((fn_off & on_pred).sum().item())
            transitions["FP_to_TN"] += int((fp_off & (~on_pred)).sum().item())
            transitions["TN_to_FP"] += int((tn_off & on_pred).sum().item())
            transitions["TP_to_FN"] += int((tp_off & (~on_pred)).sum().item())

            delta_abs_values.append(delta.detach().abs().reshape(-1).cpu().numpy())
            delta_values.append(delta.detach().reshape(-1).cpu().numpy())
            off_std_values.append(float(off_logits.detach().float().std(unbiased=False).cpu().item()))

    off_precision, off_recall, off_iou, off_f1 = summarize_counts(off_total)
    on_precision, on_recall, on_iou, on_f1 = summarize_counts(on_total)
    delta_abs = np.concatenate(delta_abs_values, axis=0) if delta_abs_values else np.empty((0,), dtype=np.float32)
    delta_signed = np.concatenate(delta_values, axis=0) if delta_values else np.empty((0,), dtype=np.float32)
    delta_abs_mean = float(delta_abs.mean()) if delta_abs.size else float("nan")
    delta_p95 = float(np.quantile(delta_abs, 0.95)) if delta_abs.size else float("nan")
    delta_max = float(delta_abs.max()) if delta_abs.size else float("nan")
    delta_mean = float(delta_signed.mean()) if delta_signed.size else float("nan")
    off_std_mean = float(np.mean(off_std_values)) if off_std_values else float("nan")
    r_boundary = delta_abs_mean / max(off_std_mean, 1e-12)

    good = transitions["FN_to_TP"] + transitions["FP_to_TN"]
    bad = transitions["TN_to_FP"] + transitions["TP_to_FN"]
    q_boundary = good / max(transitions["changed"], 1)

    print("\nBOUNDARY DELTA QUADRANT DIAGNOSTIC")
    print(f"split={args.split} batches={batches} images={images} threshold={args.threshold:.3f}")
    print(f"model_path={args.model_path}")
    print(f"guided_head.beta checkpoint value={beta_value:.8f}")
    print("")
    print("Surface metrics")
    print(
        "  Boundary OFF beta=0: IoU={:.6f} F1={:.6f} Precision={:.6f} Recall={:.6f} "
        "TP={} FP={} FN={} TN={}".format(
            off_iou,
            off_f1,
            off_precision,
            off_recall,
            off_total["tp"],
            off_total["fp"],
            off_total["fn"],
            off_total["tn"],
        )
    )
    print(
        "  Boundary ON current: IoU={:.6f} F1={:.6f} Precision={:.6f} Recall={:.6f} "
        "TP={} FP={} FN={} TN={}".format(
            on_iou,
            on_f1,
            on_precision,
            on_recall,
            on_total["tp"],
            on_total["fp"],
            on_total["fn"],
            on_total["tn"],
        )
    )
    print("")
    print("Boundary delta magnitude")
    print(f"  boundary_delta_mean:      {delta_mean:.8f}")
    print(f"  boundary_delta_abs_mean:  {delta_abs_mean:.8f}")
    print(f"  boundary_delta_p95:       {delta_p95:.8f}")
    print(f"  boundary_delta_max:       {delta_max:.8f}")
    print(f"  surface_off_logit_std:    {off_std_mean:.8f}")
    print(f"  R_boundary:               {r_boundary:.8f}")
    print("")
    print("Binary transitions: Boundary ON relative to OFF")
    print(f"  FN -> TP: {transitions['FN_to_TP']}")
    print(f"  FP -> TN: {transitions['FP_to_TN']}")
    print(f"  TN -> FP: {transitions['TN_to_FP']}")
    print(f"  TP -> FN: {transitions['TP_to_FN']}")
    print(f"  changed:  {transitions['changed']}")
    print(f"  good:     {good}")
    print(f"  bad:      {bad}")
    print(f"  Q_b:      {q_boundary:.8f}")


if __name__ == "__main__":
    main()
