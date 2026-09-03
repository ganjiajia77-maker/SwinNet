import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_structure_supervision import load_model
from datasets.dataset_road_skeleton import RoadSkeletonDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--off_mode", type=str, default="refine", choices=["refine", "refine_highres"])
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--cfg", type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--dataset", type=str, default="ImageData")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache_mode", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--accumulation_steps", type=int, default=0)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--amp_opt_level", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument("--stage_topology_stages", type=str, default="none")
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    parser.add_argument("--stage_topology_bias_mode", type=str, default="pairwise_skeleton")
    parser.add_argument("--stage_topology_ratio", type=float, default=0.08)
    parser.add_argument("--stage_topology_topo_clip", type=float, default=4.0)
    parser.add_argument("--stage2_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--stage3_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--final_skeleton_gradient_ratio", type=float, default=0.0)
    parser.add_argument("--structure_profile", type=str, default="stage23_boundary_0626")
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23")
    parser.add_argument("--highres_structure_fusion_mode", type=str, default="stage23")
    parser.add_argument("--model_impl", type=str, default="auto", choices=["auto", "standard", "selective"])
    return parser.parse_args()


def surface_metrics(prob, gt, threshold):
    pred = prob >= float(threshold)
    gt = gt.bool()
    tp = torch.logical_and(pred, gt).sum().item()
    fp = torch.logical_and(pred, ~gt).sum().item()
    fn = torch.logical_and(~pred, gt).sum().item()
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return iou, f1, precision, recall, pred


def set_structure_refine_enabled(model, enabled):
    saved = []
    for module in model.modules():
        if hasattr(module, "enable_direct_feature_refinement"):
            saved.append((module, bool(module.enable_direct_feature_refinement)))
            module.enable_direct_feature_refinement = bool(enabled)
    return saved


def restore_structure_refine(saved):
    for module, value in saved:
        module.enable_direct_feature_refinement = value


def set_highres_enabled(model, enabled):
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    if not hasattr(swin, "enable_highres_structure_stream"):
        return None
    old_value = bool(swin.enable_highres_structure_stream)
    swin.enable_highres_structure_stream = bool(enabled)
    return old_value


def restore_highres(model, old_value):
    if old_value is None:
        return
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    swin.enable_highres_structure_stream = bool(old_value)


def evaluate(model, loader, device, threshold, max_batches, desc):
    tp = fp = fn = 0.0
    probs = []
    masks = []
    images = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=desc)):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            image = batch["image"].to(device)
            mask = batch["mask"].to(device) > 0.5
            output = model(image, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
            prob = torch.sigmoid(output[0])
            if mask.shape[-2:] != prob.shape[-2:]:
                mask = F.interpolate(mask.float(), size=prob.shape[-2:], mode="nearest") > 0.5
            iou, f1, precision, recall, _ = surface_metrics(prob, mask, threshold)
            batch_pixels = mask.numel()
            # Convert batch metrics back to weighted counts through direct recount.
            pred = prob >= float(threshold)
            tp += torch.logical_and(pred, mask).sum().item()
            fp += torch.logical_and(pred, ~mask).sum().item()
            fn += torch.logical_and(~pred, mask).sum().item()
            probs.append(prob.detach().cpu())
            masks.append(mask.detach().cpu())
            images += int(image.shape[0])
    all_prob = torch.cat(probs, dim=0)
    all_mask = torch.cat(masks, dim=0)
    iou, f1, precision, recall, pred = surface_metrics(all_prob, all_mask, threshold)
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "pred": pred,
        "mask": all_mask,
        "images": images,
    }


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    model = load_model(args, device)
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
        pin_memory=True,
    )

    on = evaluate(model, loader, device, args.threshold, args.max_batches, "Structure ON")
    saved_refine = set_structure_refine_enabled(model, False)
    saved_highres = None
    if args.off_mode == "refine_highres":
        saved_highres = set_highres_enabled(model, False)
    off = evaluate(model, loader, device, args.threshold, args.max_batches, "Structure OFF")
    restore_highres(model, saved_highres)
    restore_structure_refine(saved_refine)

    print(f"Checkpoint: {args.model_path}")
    print(f"split={args.split}, threshold={args.threshold}, off_mode={args.off_mode}, images={on['images']}")
    print("\nStructure ON/OFF")
    print(f"  OFF: IoU={off['iou']:.4f}, F1={off['f1']:.4f}, P={off['precision']:.4f}, R={off['recall']:.4f}")
    print(f"  ON:  IoU={on['iou']:.4f}, F1={on['f1']:.4f}, P={on['precision']:.4f}, R={on['recall']:.4f}")
    print(
        "  Delta(ON-OFF): "
        f"IoU={on['iou'] - off['iou']:+.4f}, "
        f"F1={on['f1'] - off['f1']:+.4f}, "
        f"P={on['precision'] - off['precision']:+.4f}, "
        f"R={on['recall'] - off['recall']:+.4f}"
    )

    gt = on["mask"]
    on_pred = on["pred"]
    off_pred = off["pred"]
    print("\nChanged Pixels")
    print(f"  OFF FN -> ON TP: {int(((~off_pred) & on_pred & gt).sum().item())}")
    print(f"  OFF TN -> ON FP: {int(((~off_pred) & on_pred & (~gt)).sum().item())}")
    print(f"  OFF FP -> ON TN: {int((off_pred & (~on_pred) & (~gt)).sum().item())}")
    print(f"  OFF TP -> ON FN: {int((off_pred & (~on_pred) & gt).sum().item())}")


if __name__ == "__main__":
    main()
