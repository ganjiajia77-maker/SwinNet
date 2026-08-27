import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import (
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
    print_topology_coefficients,
)
from networks.vision_transformer_selective_fusion import (
    SwinUnet as ViT_seg_selective,
    load_topology_checkpoint_state as load_topology_checkpoint_state_selective,
    print_topology_coefficients as print_topology_coefficients_selective,
)
from networks.skeleton_guided_head import (
    LegacyConvConnectivityHead,
    LegacyPairwisePriorConnectivityHead,
)
from networks.skeleton_guided_head_selective_fusion import (
    LegacyConvConnectivityHead as LegacyConvConnectivityHeadSelective,
)
from topology_direction_constants import CONNECTIVITY_DIR_NAMES, CONNECTIVITY_DIRECTIONS


DIR_NAMES = list(CONNECTIVITY_DIR_NAMES)
DIR_OFFSETS = torch.tensor(CONNECTIVITY_DIRECTIONS, dtype=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=r"D:\Code\Massachusetts\tiff")
    parser.add_argument(
        "--model_path",
        type=str,
        default=(
            r"D:\Code\Swin-Unet-main\model_out"
            r"\ma_baseline_bcedice_w008012_gr05_aux1024_swinv2_direct512_bs1_acc4_20260812"
            r"\best.pth"
        ),
    )
    parser.add_argument("--output_dir", type=str, default="./analysis_out/structure_supervision_512")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--cfg", type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--dataset", type=str, default="ImageData")
    parser.add_argument("--n_class", type=int, default=2)
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
    parser.add_argument("--structure_profile", type=str, default=STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626)
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
    parser.add_argument("--stage2_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--stage3_skeleton_gradient_ratio", type=float, default=0.5)
    parser.add_argument("--final_skeleton_gradient_ratio", type=float, default=0.0)
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23")
    parser.add_argument("--highres_structure_fusion_mode", type=str, default="stage23")
    parser.add_argument(
        "--model_impl",
        type=str,
        default="auto",
        choices=["auto", "standard", "selective"],
        help="model implementation used by the checkpoint",
    )
    return parser.parse_args()


def resize_like(x, reference, mode="nearest"):
    if x.shape[-2:] == reference.shape[-2:]:
        return x
    kwargs = {"size": reference.shape[-2:], "mode": mode}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    return F.interpolate(x.float(), **kwargs)


def select_stage_outputs(structure_outputs):
    selected = {}
    for item in structure_outputs:
        if "connectivity" not in item or "direction" not in item:
            continue
        stage = item.get("stage")
        step = item.get("refinement_step", None)
        if stage in (2, 3) and step == 1:
            selected[f"stage{stage}_refine"] = item
    for item in structure_outputs:
        stage = item.get("stage")
        if stage in (2, 3) and f"stage{stage}_refine" not in selected:
            selected[f"stage{stage}_last"] = item
    return selected


def adapt_connectivity_modules_for_checkpoint(model, state_dict, model_impl):
    keys = [str(key) for key in state_dict.keys()]
    has_pairwise_head = any(".connectivity_head.edge_mlp." in key for key in keys)
    has_legacy_conv_head = any(key.endswith(".connectivity_head.weight") for key in keys)
    has_connectivity_context = any(".connectivity_context." in key for key in keys)

    if not has_connectivity_context:
        for module in model.modules():
            if hasattr(module, "connectivity_context"):
                module.connectivity_context = torch.nn.Identity()

    pairwise_prior_replaced = 0
    if has_pairwise_head and model_impl != "selective":
        divisor = 4 if model_impl == "selective" else 2
        for module_name, module in model.named_modules():
            head = getattr(module, "connectivity_head", None)
            if head is None or not hasattr(head, "edge_mlp"):
                continue
            key = f"{module_name}.connectivity_head.edge_mlp.0.weight"
            checkpoint_weight = state_dict.get(key)
            if checkpoint_weight is None:
                continue
            first = head.edge_mlp[0]
            if int(checkpoint_weight.shape[1]) == int(first.in_channels):
                continue
            channels = int(getattr(head, "feature_channels", 0))
            if channels <= 0:
                prior_channels = int(getattr(head, "prior_channels", 16))
                channels = (int(first.in_channels) - prior_channels) // divisor
            legacy_in_channels = divisor * channels + 3
            if int(checkpoint_weight.shape[1]) != legacy_in_channels:
                continue
            connectivity_channels = getattr(head, "connectivity_channels", 8)
            hidden_channels = int(checkpoint_weight.shape[0])
            device = first.weight.device
            dtype = first.weight.dtype
            module.connectivity_head = LegacyPairwisePriorConnectivityHead(
                channels,
                connectivity_channels,
                hidden_channels=hidden_channels,
            ).to(device=device, dtype=dtype)
            pairwise_prior_replaced += 1
        if pairwise_prior_replaced:
            print(
                f"[INFO] Checkpoint uses legacy pairwise prior connectivity heads "
                f"(2C+3); replaced {pairwise_prior_replaced} heads for compatible evaluation.",
                flush=True,
            )

    if not has_legacy_conv_head or has_pairwise_head:
        return

    legacy_cls = (
        LegacyConvConnectivityHeadSelective
        if model_impl == "selective"
        else LegacyConvConnectivityHead
    )
    divisor = 4 if model_impl == "selective" else 2
    replaced = 0
    for module in model.modules():
        head = getattr(module, "connectivity_head", None)
        if head is None or not hasattr(head, "edge_mlp"):
            continue
        first = head.edge_mlp[0]
        channels = int(getattr(head, "feature_channels", 0))
        if channels <= 0:
            prior_channels = int(getattr(head, "prior_channels", 3))
            channels = (int(first.in_channels) - prior_channels) // divisor
        connectivity_channels = getattr(head, "connectivity_channels", 8)
        device = first.weight.device
        dtype = first.weight.dtype
        module.connectivity_head = legacy_cls(channels, connectivity_channels).to(
            device=device,
            dtype=dtype,
        )
        replaced += 1
    if replaced:
        print(
            f"[INFO] Checkpoint uses legacy conv connectivity heads; "
            f"replaced {replaced} pairwise heads for compatible evaluation.",
            flush=True,
        )


def load_model(args, device):
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if isinstance(saved_args, dict):
        args.structure_profile = saved_args.get("structure_profile", args.structure_profile)
        args.disable_msfe_skip = bool(saved_args.get("disable_msfe_skip", args.disable_msfe_skip))
        for name in (
            "stage_topology_stages",
            "stage_topology_alpha_max",
            "stage_topology_alpha_init",
            "stage_topology_bias_mode",
            "stage_topology_ratio",
            "stage_topology_topo_clip",
            "stage2_skeleton_gradient_ratio",
            "stage3_skeleton_gradient_ratio",
            "final_skeleton_gradient_ratio",
            "bottleneck_type",
            "enable_highres_structure_stream",
            "highres_structure_channels",
            "highres_structure_fuse_stages",
            "highres_structure_fusion_mode",
        ):
            if name in saved_args:
                setattr(args, name, saved_args[name])
    if checkpoint.get("structure_profile"):
        args.structure_profile = checkpoint["structure_profile"]

    config = get_config(args)
    model_impl = getattr(args, "model_impl", "auto")
    if model_impl == "auto":
        probe = state_dict.get(
            "swin_unet.decoder_structure_blocks.2.connectivity_head.edge_mlp.0.weight"
        )
        if probe is not None and probe.dim() == 4:
            # Standard pairwise head uses [feature, neighbor, prior_embed] -> 2C+P.
            # In these blocks hidden_channels is C/2, so input is 4*out+P.
            # Selective-fusion head adds [feature-neighbor, feature*neighbor] -> 4C+P,
            # which is 8*out+P for the same hidden width. Older checkpoints used P=3.
            out_channels = int(probe.shape[0])
            in_channels = int(probe.shape[1])
            if in_channels in (4 * out_channels + 3, 4 * out_channels + 16):
                model_impl = "standard"
            elif in_channels in (8 * out_channels + 3, 8 * out_channels + 16):
                model_impl = "selective"
            else:
                model_impl = "selective" if "selective" in str(saved_args.get("run_name", "")).lower() else "standard"
        else:
            model_impl = "selective" if "selective" in str(saved_args.get("run_name", "")).lower() else "standard"
    model_cls = ViT_seg_selective if model_impl == "selective" else ViT_seg
    loader = (
        load_topology_checkpoint_state_selective
        if model_impl == "selective"
        else load_topology_checkpoint_state
    )
    printer = (
        print_topology_coefficients_selective
        if model_impl == "selective"
        else print_topology_coefficients
    )
    print(f"[INFO] Diagnostic model implementation: {model_impl}", flush=True)

    model = model_cls(
        config=config,
        img_size=args.img_size,
        num_classes=args.num_classes,
        return_skeleton=True,
        bottleneck_type=args.bottleneck_type,
        final_topology_eta_init=args.final_topology_eta_init,
        final_gap_rho_init=args.final_gap_rho_init,
        stage_topology_stages=args.stage_topology_stages,
        stage_topology_alpha_max=args.stage_topology_alpha_max,
        stage_topology_alpha_init=args.stage_topology_alpha_init,
        stage_topology_bias_mode=args.stage_topology_bias_mode,
        stage_topology_ratio=args.stage_topology_ratio,
        stage_topology_topo_clip=args.stage_topology_topo_clip,
        structure_profile=args.structure_profile,
        use_msfe_skip=not args.disable_msfe_skip,
        stage2_skeleton_gradient_ratio=args.stage2_skeleton_gradient_ratio,
        stage3_skeleton_gradient_ratio=args.stage3_skeleton_gradient_ratio,
        final_skeleton_gradient_ratio=args.final_skeleton_gradient_ratio,
        enable_highres_structure_stream=args.enable_highres_structure_stream,
        highres_structure_channels=args.highres_structure_channels,
        highres_structure_fuse_stages=args.highres_structure_fuse_stages,
        highres_structure_fusion_mode=args.highres_structure_fusion_mode,
    )
    adapt_connectivity_modules_for_checkpoint(model, state_dict, model_impl)
    loader(
        model,
        state_dict,
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=(args.bottleneck_type == "global_local"),
    )
    model.to(device).eval()
    printer(model)
    return model


class ConnectivityStats:
    def __init__(self):
        self.rows = {}
        self.hist_bins = torch.linspace(0.0, 1.0, 11)

    def _state(self, key):
        if key not in self.rows:
            self.rows[key] = {
                "gt_pos": torch.zeros(8, dtype=torch.float64),
                "gt_neg": torch.zeros(8, dtype=torch.float64),
                "tp": torch.zeros(8, dtype=torch.float64),
                "fp": torch.zeros(8, dtype=torch.float64),
                "fn": torch.zeros(8, dtype=torch.float64),
                "pos_prob_sum": torch.zeros(8, dtype=torch.float64),
                "neg_prob_sum": torch.zeros(8, dtype=torch.float64),
                "pos_prob_sq": torch.zeros(8, dtype=torch.float64),
                "neg_prob_sq": torch.zeros(8, dtype=torch.float64),
                "pos_hist": torch.zeros(8, 10, dtype=torch.float64),
                "neg_hist": torch.zeros(8, 10, dtype=torch.float64),
            }
        return self.rows[key]

    def add(self, key, prob, gt, valid):
        state = self._state(key)
        valid = valid.bool().expand_as(gt)
        pred = (prob >= 0.5) & valid
        gt_bool = (gt >= 0.5) & valid
        neg_bool = (~gt_bool) & valid
        for direction in range(8):
            p = prob[:, direction][valid[:, direction]].detach().cpu().float()
            g = gt_bool[:, direction][valid[:, direction]].detach().cpu()
            pred_d = pred[:, direction][valid[:, direction]].detach().cpu()
            pos = g
            neg = ~g
            state["gt_pos"][direction] += pos.sum()
            state["gt_neg"][direction] += neg.sum()
            state["tp"][direction] += (pred_d & pos).sum()
            state["fp"][direction] += (pred_d & neg).sum()
            state["fn"][direction] += ((~pred_d) & pos).sum()
            if pos.any():
                pos_p = p[pos]
                state["pos_prob_sum"][direction] += pos_p.double().sum()
                state["pos_prob_sq"][direction] += (pos_p.double() ** 2).sum()
                state["pos_hist"][direction] += torch.histc(pos_p, bins=10, min=0.0, max=1.0).double()
            if neg.any():
                neg_p = p[neg]
                state["neg_prob_sum"][direction] += neg_p.double().sum()
                state["neg_prob_sq"][direction] += (neg_p.double() ** 2).sum()
                state["neg_hist"][direction] += torch.histc(neg_p, bins=10, min=0.0, max=1.0).double()

    @staticmethod
    def _mean_std(sum_value, sq_value, count):
        if count <= 0:
            return 0.0, 0.0
        mean = float(sum_value / count)
        var = max(float(sq_value / count) - mean * mean, 0.0)
        return mean, var ** 0.5

    def write(self, path):
        fields = [
            "stage",
            "scope",
            "direction",
            "gt_pos",
            "gt_neg",
            "pos_ratio",
            "precision",
            "recall",
            "f1",
            "pos_prob_mean",
            "pos_prob_std",
            "neg_prob_mean",
            "neg_prob_std",
            "pos_hist_0_0.1",
            "pos_hist_0.1_0.2",
            "pos_hist_0.2_0.3",
            "pos_hist_0.3_0.4",
            "pos_hist_0.4_0.5",
            "pos_hist_0.5_0.6",
            "pos_hist_0.6_0.7",
            "pos_hist_0.7_0.8",
            "pos_hist_0.8_0.9",
            "pos_hist_0.9_1.0",
            "neg_hist_0_0.1",
            "neg_hist_0.1_0.2",
            "neg_hist_0.2_0.3",
            "neg_hist_0.3_0.4",
            "neg_hist_0.4_0.5",
            "neg_hist_0.5_0.6",
            "neg_hist_0.6_0.7",
            "neg_hist_0.7_0.8",
            "neg_hist_0.8_0.9",
            "neg_hist_0.9_1.0",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for key, state in sorted(self.rows.items()):
                stage, scope = key.split("|", 1)
                for direction in range(8):
                    gt_pos = float(state["gt_pos"][direction])
                    gt_neg = float(state["gt_neg"][direction])
                    tp = float(state["tp"][direction])
                    fp = float(state["fp"][direction])
                    fn = float(state["fn"][direction])
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
                    pos_mean, pos_std = self._mean_std(
                        state["pos_prob_sum"][direction],
                        state["pos_prob_sq"][direction],
                        gt_pos,
                    )
                    neg_mean, neg_std = self._mean_std(
                        state["neg_prob_sum"][direction],
                        state["neg_prob_sq"][direction],
                        gt_neg,
                    )
                    row = {
                        "stage": stage,
                        "scope": scope,
                        "direction": DIR_NAMES[direction],
                        "gt_pos": int(gt_pos),
                        "gt_neg": int(gt_neg),
                        "pos_ratio": gt_pos / (gt_pos + gt_neg + 1e-8),
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "pos_prob_mean": pos_mean,
                        "pos_prob_std": pos_std,
                        "neg_prob_mean": neg_mean,
                        "neg_prob_std": neg_std,
                    }
                    for idx, value in enumerate(state["pos_hist"][direction].tolist()):
                        row[fields[13 + idx]] = int(value)
                    for idx, value in enumerate(state["neg_hist"][direction].tolist()):
                        row[fields[23 + idx]] = int(value)
                    writer.writerow(row)


class DirectionStats:
    def __init__(self):
        self.rows = {}

    def _state(self, key):
        if key not in self.rows:
            self.rows[key] = {
                "count": 0.0,
                "correct": 0.0,
                "angle_sum": 0.0,
                "angle_sq": 0.0,
                "entropy_sum": 0.0,
                "pred_hist": torch.zeros(8, dtype=torch.float64),
                "gt_hist": torch.zeros(8, dtype=torch.float64),
            }
        return self.rows[key]

    def add(self, key, direction_logits, direction_gt, valid):
        state = self._state(key)
        pred_vec = F.normalize(direction_logits.float(), dim=1, eps=1e-6)
        gt_vec = F.normalize(direction_gt.float(), dim=1, eps=1e-6)
        offsets = DIR_OFFSETS.to(direction_logits.device, direction_logits.dtype)
        offsets = F.normalize(offsets, dim=1, eps=1e-6)
        pred_score = torch.einsum("bchw,kc->bkhw", pred_vec, offsets)
        gt_score = torch.einsum("bchw,kc->bkhw", gt_vec, offsets)
        pred_idx = pred_score.argmax(dim=1)
        gt_idx = gt_score.argmax(dim=1)
        prob8 = torch.softmax(pred_score * 8.0, dim=1)
        entropy = -(prob8 * (prob8 + 1e-8).log()).sum(dim=1)
        cosine = (pred_vec * gt_vec).sum(dim=1).clamp(-1.0, 1.0)
        angle = torch.rad2deg(torch.acos(cosine))
        mask = valid.squeeze(1).bool()
        if mask.sum() == 0:
            return
        pred_flat = pred_idx[mask].detach().cpu()
        gt_flat = gt_idx[mask].detach().cpu()
        angle_flat = angle[mask].detach().cpu().double()
        entropy_flat = entropy[mask].detach().cpu().double()
        state["count"] += float(mask.sum().item())
        state["correct"] += float((pred_flat == gt_flat).sum().item())
        state["angle_sum"] += float(angle_flat.sum().item())
        state["angle_sq"] += float((angle_flat ** 2).sum().item())
        state["entropy_sum"] += float(entropy_flat.sum().item())
        state["pred_hist"] += torch.bincount(pred_flat, minlength=8).double()
        state["gt_hist"] += torch.bincount(gt_flat, minlength=8).double()

    def write(self, path):
        fields = [
            "stage",
            "scope",
            "pixels",
            "accuracy_8dir",
            "angle_error_mean_deg",
            "angle_error_std_deg",
            "entropy_mean",
        ] + [f"pred_{name}" for name in DIR_NAMES] + [f"gt_{name}" for name in DIR_NAMES]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for key, state in sorted(self.rows.items()):
                stage, scope = key.split("|", 1)
                count = state["count"]
                angle_mean = state["angle_sum"] / (count + 1e-8)
                angle_var = max(state["angle_sq"] / (count + 1e-8) - angle_mean ** 2, 0.0)
                row = {
                    "stage": stage,
                    "scope": scope,
                    "pixels": int(count),
                    "accuracy_8dir": state["correct"] / (count + 1e-8),
                    "angle_error_mean_deg": angle_mean,
                    "angle_error_std_deg": angle_var ** 0.5,
                    "entropy_mean": state["entropy_sum"] / (count + 1e-8),
                }
                for idx, name in enumerate(DIR_NAMES):
                    row[f"pred_{name}"] = int(state["pred_hist"][idx].item())
                    row[f"gt_{name}"] = int(state["gt_hist"][idx].item())
                writer.writerow(row)


def update_break_buckets(summary, break_mask):
    buckets = [
        ("<=2px", 1, 2),
        ("2-4px", 3, 4),
        ("4-8px", 5, 8),
        ("8-16px", 9, 16),
        (">16px", 17, 10**9),
    ]
    for image_idx in range(break_mask.shape[0]):
        binary = break_mask[image_idx, 0].detach().cpu().numpy().astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        for label in range(1, num_labels):
            size = int(stats[label, cv2.CC_STAT_AREA])
            for name, low, high in buckets:
                if low <= size <= high:
                    summary[name]["components"] += 1
                    summary[name]["pixels"] += size
                    break


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    c_stats = ConnectivityStats()
    d_stats = DirectionStats()
    break_summary = {
        "<=2px": {"components": 0, "pixels": 0},
        "2-4px": {"components": 0, "pixels": 0},
        "4-8px": {"components": 0, "pixels": 0},
        "8-16px": {"components": 0, "pixels": 0},
        ">16px": {"components": 0, "pixels": 0},
    }
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images = batch["image"].to(device)
            masks = (batch["mask"].to(device) > 0.5)
            skeleton_raw = batch["skeleton"].to(device).float()
            skeleton_dilate_raw = batch["skeleton_dilate"].to(device).float()
            connectivity_gt = batch["connectivity_gt"].to(device).float()
            direction_gt = batch["direction_gt"].to(device).float()

            outputs = model(images, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
            surface_logits = outputs[0]
            skeleton = resize_like(skeleton_raw, surface_logits, mode="nearest") > 0.5
            skeleton_dilate = resize_like(
                skeleton_dilate_raw,
                surface_logits,
                mode="nearest",
            ) > 0.5
            surface_pred = torch.sigmoid(surface_logits) >= args.threshold
            tp += int((surface_pred & masks).sum().item())
            fp += int((surface_pred & (~masks)).sum().item())
            fn += int(((~surface_pred) & masks).sum().item())
            tn += int(((~surface_pred) & (~masks)).sum().item())
            break_mask = skeleton & masks & (~surface_pred)
            update_break_buckets(break_summary, break_mask)

            for stage_name, stage_output in select_stage_outputs(outputs[-1]).items():
                c_prob = torch.sigmoid(stage_output["connectivity"])
                c_gt = resize_like(connectivity_gt, c_prob, mode="nearest")
                skel_stage = resize_like(skeleton.float(), c_prob[:, :1], mode="nearest") > 0.5
                skel_dilate_stage = resize_like(
                    skeleton_dilate.float(),
                    c_prob[:, :1],
                    mode="nearest",
                ) > 0.5
                valid_all = torch.ones_like(c_prob[:, :1], dtype=torch.bool)
                c_stats.add(f"{stage_name}|all", c_prob, c_gt, valid_all)
                c_stats.add(f"{stage_name}|skeleton_dilate", c_prob, c_gt, skel_dilate_stage)

                d_logits = stage_output["direction"]
                d_gt = resize_like(direction_gt, d_logits[:, :1], mode="nearest")
                d_stats.add(f"{stage_name}|skeleton", d_logits, d_gt, skel_stage)
                d_stats.add(f"{stage_name}|skeleton_dilate", d_logits, d_gt, skel_dilate_stage)

            if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == len(loader):
                print(
                    f"{batch_idx + 1}/{len(loader)} TP={tp} FP={fp} FN={fn} TN={tn}",
                    flush=True,
                )

    c_path = os.path.join(args.output_dir, "connectivity_direction_metrics.csv")
    d_path = os.path.join(args.output_dir, "direction_metrics.csv")
    b_path = os.path.join(args.output_dir, "break_gap_length_buckets.csv")
    c_stats.write(c_path)
    d_stats.write(d_path)
    with open(b_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["bucket", "components", "pixels"])
        writer.writeheader()
        for bucket, values in break_summary.items():
            writer.writerow(
                {
                    "bucket": bucket,
                    "components": values["components"],
                    "pixels": values["pixels"],
                }
            )

    print("\nSurface confusion")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print("\nBreak gap buckets")
    for bucket, values in break_summary.items():
        print(f"{bucket}: components={values['components']} pixels={values['pixels']}")
    print(f"\nSaved: {c_path}")
    print(f"Saved: {d_path}")
    print(f"Saved: {b_path}")


if __name__ == "__main__":
    main()
