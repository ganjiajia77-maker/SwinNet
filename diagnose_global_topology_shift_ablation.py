import argparse
import copy
import csv
import os
import sys
import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_structure_supervision import adapt_connectivity_modules_for_checkpoint
from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import binary_metrics_from_logits
from networks.vision_transformer import (
    STRUCTURE_PROFILE_FULL,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_FINAL_SKE,
    SwinUnet,
    load_topology_checkpoint_state,
)


MODES = ("baseline", "z_struct", "z_struct_connectivity", "z_struct_direction", "full")


def _cli_has(name):
    return name in sys.argv[1:]


def inherit_checkpoint_args(args, checkpoint):
    saved_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    if not isinstance(saved_args, dict):
        return
    for name in (
        "bottleneck_type",
        "structure_profile",
        "disable_msfe_skip",
        "stage_topology_stages",
        "stage_topology_alpha_max",
        "stage_topology_alpha_init",
        "enable_highres_structure_stream",
        "highres_structure_channels",
        "highres_structure_fuse_stages",
        "highres_structure_fusion_mode",
        "enable_post_refine_structure_interaction",
        "global_topology_max_nodes",
        "global_topology_heads",
        "global_topology_alpha_max",
    ):
        flag = "--" + name.replace("_", "-")
        flag_underscore = "--" + name
        if name in saved_args and not (_cli_has(flag) or _cli_has(flag_underscore)):
            setattr(args, name, saved_args[name])
    if "enable_global_topology" in saved_args and not _cli_has("--enable_global_topology"):
        args.enable_global_topology = bool(saved_args["enable_global_topology"])


def build_model(args, checkpoint, enable_global_topology):
    model_args = copy.copy(args)
    model_args.enable_global_topology = bool(enable_global_topology)
    config = get_config(model_args)
    model = SwinUnet(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        return_skeleton=True,
        bottleneck_type=args.bottleneck_type,
        final_topology_eta_init=args.final_topology_eta_init,
        final_gap_rho_init=args.final_gap_rho_init,
        stage_topology_stages=args.stage_topology_stages,
        stage_topology_alpha_max=args.stage_topology_alpha_max,
        stage_topology_alpha_init=args.stage_topology_alpha_init,
        structure_profile=args.structure_profile,
        use_msfe_skip=not args.disable_msfe_skip,
        enable_highres_structure_stream=args.enable_highres_structure_stream,
        highres_structure_channels=args.highres_structure_channels,
        highres_structure_fuse_stages=args.highres_structure_fuse_stages,
        highres_structure_fusion_mode=args.highres_structure_fusion_mode,
        enable_post_refine_structure_interaction=(
            args.enable_post_refine_structure_interaction
        ),
        enable_global_topology=enable_global_topology,
        global_topology_max_nodes=args.global_topology_max_nodes,
        global_topology_heads=args.global_topology_heads,
        global_topology_alpha_max=args.global_topology_alpha_max,
    )
    state_dict = checkpoint["model_state_dict"]
    adapt_connectivity_modules_for_checkpoint(model, state_dict, "standard")
    load_topology_checkpoint_state(
        model,
        state_dict,
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
    )
    return model


def force_global_topology_mode(model, mode):
    module = model.module if hasattr(model, "module") else model
    swin = module.swin_unet
    gt = getattr(swin, "global_topology", None)
    if gt is None:
        raise RuntimeError("Model has no global_topology module.")
    if mode == "baseline":
        swin.enable_global_topology = False
        gt.enable_global_topology = False
        return gt

    swin.enable_global_topology = True
    gt.enable_global_topology = True
    gt._diagnose_mode = mode

    def ablated_forward(
        self,
        feature,
        z_struct,
        surface_prob,
        connectivity_feature=None,
        direction_feature=None,
    ):
        batch, channels, height, width = feature.shape
        if not self.enable_global_topology or z_struct is None or surface_prob is None:
            return feature
        with torch.no_grad():
            surface_gate = surface_prob.detach().to(dtype=feature.dtype)
            if surface_gate.shape[-2:] != (height, width):
                surface_gate = F.interpolate(
                    surface_gate,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            surface_gate_for_anchor = surface_gate.clamp(0.0, 1.0)
            surface_gate_for_residual = surface_gate.clamp(0.0, 1.0)

        with torch.no_grad():
            z_score = torch.linalg.vector_norm(
                z_struct.detach().float(),
                dim=1,
                keepdim=True,
            )
            z_score = self._minmax_normalize_map(z_score).to(dtype=feature.dtype)
            if z_score.shape[-2:] != (height, width):
                z_score = F.interpolate(
                    z_score,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            anchor_score = z_score * surface_gate_for_anchor
            coords, valid, scores, candidate_count = self._extract_fps_anchors(
                anchor_score
            )

        sampled_struct = self._sample_features_at_anchor_coords(
            z_struct.to(dtype=feature.dtype),
            coords,
            anchor_hw=(height, width),
        )
        sampled_feature = self._sample_features_at_anchor_coords(
            feature,
            coords,
            anchor_hw=(height, width),
        )
        connectivity_map = self._prepare_token_map(
            connectivity_feature,
            batch,
            (height, width),
            self.connectivity_channels,
            feature.dtype,
            feature.device,
        )
        direction_map = self._prepare_token_map(
            direction_feature,
            batch,
            (height, width),
            self.direction_channels,
            feature.dtype,
            feature.device,
        )
        sampled_connectivity = self._sample_features_at_anchor_coords(
            connectivity_map,
            coords,
            anchor_hw=(height, width),
        )
        sampled_direction = self._sample_features_at_anchor_coords(
            direction_map,
            coords,
            anchor_hw=(height, width),
        )
        if mode == "z_struct":
            sampled_feature = torch.zeros_like(sampled_feature)
            sampled_connectivity = torch.zeros_like(sampled_connectivity)
            sampled_direction = torch.zeros_like(sampled_direction)
        elif mode == "z_struct_connectivity":
            sampled_feature = torch.zeros_like(sampled_feature)
            sampled_direction = torch.zeros_like(sampled_direction)
        elif mode == "z_struct_direction":
            sampled_feature = torch.zeros_like(sampled_feature)
            sampled_connectivity = torch.zeros_like(sampled_connectivity)
        elif mode == "full":
            pass
        else:
            raise RuntimeError(f"Unknown ablation mode: {mode}")
        node_types = torch.zeros(
            batch,
            self.max_nodes,
            device=feature.device,
            dtype=torch.long,
        )
        coords_norm = coords.float() / feature.new_tensor(
            [max(height - 1, 1), max(width - 1, 1)]
        )
        node_input = torch.cat(
            [
                sampled_struct,
                sampled_feature,
                sampled_connectivity,
                sampled_direction,
                self.node_type_embedding(node_types),
                coords_norm,
            ],
            dim=-1,
        )
        node_feature = self.node_projection(node_input)
        if hasattr(self, "_refine_tokens_with_relative_topology"):
            node_feature, _ = self._refine_tokens_with_relative_topology(
                node_feature,
                coords,
                valid,
                anchor_hw=(height, width),
            )
        grid_tokens = feature.flatten(2).transpose(1, 2)
        query = self.grid_q(grid_tokens).reshape(
            batch,
            height * width,
            self.heads,
            channels // self.heads,
        ).permute(0, 2, 1, 3)
        node_kv = self.node_kv(node_feature).reshape(
            batch,
            self.max_nodes,
            2,
            self.heads,
            channels // self.heads,
        ).permute(2, 0, 3, 1, 4)
        key, value = node_kv[0], node_kv[1]
        logits = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(
            channels // self.heads
        )
        logits = logits.masked_fill(
            ~valid[:, None, None, :],
            -torch.finfo(logits.dtype).max,
        )
        attention = torch.softmax(logits, dim=-1)
        context = torch.matmul(attention, value).transpose(1, 2).reshape(
            batch,
            height * width,
            channels,
        )
        context = self.output_projection(context)
        context = context.transpose(1, 2).reshape(batch, channels, height, width)
        has_anchor = valid.any(dim=1).to(dtype=context.dtype).view(batch, 1, 1, 1)
        context = context * has_anchor
        delta = self.grid_projection(context)
        output = feature + self.alpha_global * delta * surface_gate_for_residual
        if self.capture_diagnostics:
            with torch.no_grad():
                valid_counts = valid.sum(dim=1).clamp_min(1)
                first_valid = torch.zeros(batch, device=valid.device, dtype=torch.long)
                for batch_index in range(batch):
                    valid_indices = torch.nonzero(valid[batch_index], as_tuple=False)
                    if valid_indices.numel() > 0:
                        first_valid[batch_index] = valid_indices[0, 0]
                batch_indices = torch.arange(batch, device=valid.device)
                attention_map = attention.mean(dim=1)[
                    batch_indices,
                    :,
                    first_valid,
                ].reshape(batch, height, width)
                self.last_diagnostics = {
                    "anchor_count": valid.sum(dim=1).float().detach(),
                    "candidate_count": feature.new_full(
                        (batch,),
                        float(candidate_count),
                    ).detach(),
                    "anchor_score_mean": scores.masked_fill(~valid, 0.0).sum(dim=1)
                    / valid.sum(dim=1).clamp_min(1).float(),
                    "anchor_score_max": scores.amax(dim=1).detach(),
                    "alpha_global": self.alpha_global.detach(),
                    "surface_gate_mean": surface_gate.mean(dim=(1, 2, 3)).detach(),
                    "surface_gate_max": surface_gate.amax(dim=(1, 2, 3)).detach(),
                    "global_residual_relative_norm": (
                        torch.linalg.vector_norm(output - feature)
                        / (torch.linalg.vector_norm(feature) + 1e-6)
                    ).detach(),
                    "feature_delta_abs_mean": (output - feature).abs().mean().detach(),
                    "attention_map": attention_map.detach().cpu(),
                    "anchor_coords": coords.detach().cpu(),
                    "anchor_valid": valid.detach().cpu(),
                    "first_valid_anchor": first_valid.detach().cpu(),
                }
        return output

    gt.forward_feature_anchors = types.MethodType(ablated_forward, gt)
    return gt


def resize_target(target, size):
    if target.shape[-2:] == size:
        return target
    return F.interpolate(target.float(), size=size, mode="nearest")


class ShiftStats:
    def __init__(self):
        self.delta_sum = {
            mode: {name: 0.0 for name in ("all", "TP", "FN", "FP", "TN")}
            for mode in MODES
            if mode != "baseline"
        }
        self.delta_count = {
            mode: {name: 0 for name in ("all", "TP", "FN", "FP", "TN")}
            for mode in MODES
            if mode != "baseline"
        }
        self.metrics = {name: [] for name in MODES}
        self.transitions = {
            name: {
                "TP_to_FN": 0,
                "FN_to_TP": 0,
                "FP_to_TN": 0,
                "TN_to_FP": 0,
                "changed": 0,
            }
            for name in MODES
            if name != "baseline"
        }

    def add_delta_region(self, mode, name, delta_abs, mask):
        count = int(mask.sum().item())
        if count == 0:
            return
        self.delta_sum[mode][name] += float(delta_abs[mask].sum().detach().cpu())
        self.delta_count[mode][name] += count

    def add_shift(self, mode, baseline_logits, logits, target, threshold):
        target = resize_target(target, logits.shape[-2:])
        baseline_pred = torch.sigmoid(baseline_logits) >= threshold
        pred = torch.sigmoid(logits) >= threshold
        truth = target >= 0.5
        delta_abs = (logits - baseline_logits).abs()
        regions = {
            "all": torch.ones_like(truth, dtype=torch.bool),
            "TP": baseline_pred & truth,
            "FN": (~baseline_pred) & truth,
            "FP": baseline_pred & (~truth),
            "TN": (~baseline_pred) & (~truth),
        }
        for name, mask in regions.items():
            self.add_delta_region(mode, name, delta_abs, mask)
        tr = self.transitions[mode]
        tr["TP_to_FN"] += int((regions["TP"] & (~pred)).sum().item())
        tr["FN_to_TP"] += int((regions["FN"] & pred).sum().item())
        tr["FP_to_TN"] += int((regions["FP"] & (~pred)).sum().item())
        tr["TN_to_FP"] += int((regions["TN"] & pred).sum().item())
        tr["changed"] += int((baseline_pred != pred).sum().item())


def mean_or_zero(values):
    if not values:
        return 0.0
    return float(np.mean(values))


def image_tensor_to_uint8(image):
    array = image.detach().cpu().float().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = np.clip((array * std + mean) * 255.0, 0, 255).astype(np.uint8)
    return array


def normalize_to_uint8(array):
    array = array.astype(np.float32)
    low = float(array.min())
    high = float(array.max())
    if high <= low + 1e-8:
        return np.zeros_like(array, dtype=np.uint8)
    return ((array - low) * 255.0 / (high - low)).astype(np.uint8)


def save_attention_visuals(output_dir, mode, batch, diagnostics, start_index, max_visuals):
    if not output_dir or start_index >= max_visuals:
        return start_index
    attention_map = diagnostics.get("attention_map")
    coords = diagnostics.get("anchor_coords")
    valid = diagnostics.get("anchor_valid")
    if attention_map is None or coords is None or valid is None:
        return start_index
    visual_dir = os.path.join(output_dir, "attention_maps", mode)
    os.makedirs(visual_dir, exist_ok=True)
    names = batch.get("case_name", batch.get("image_name", None))
    images = batch["image"]
    batch_size = images.shape[0]
    for item_index in range(batch_size):
        if start_index >= max_visuals:
            break
        rgb = image_tensor_to_uint8(images[item_index])
        heat = attention_map[item_index].numpy()
        heat = cv2.resize(
            heat,
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        heat_color = cv2.applyColorMap(normalize_to_uint8(heat), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            0.60,
            heat_color,
            0.40,
            0,
        )
        valid_coords = coords[item_index][valid[item_index]].numpy()
        anchor_h = max(int(attention_map.shape[1]) - 1, 1)
        anchor_w = max(int(attention_map.shape[2]) - 1, 1)
        for y, x in valid_coords[:64]:
            yy = int(round(float(y) * (rgb.shape[0] - 1) / anchor_h))
            xx = int(round(float(x) * (rgb.shape[1] - 1) / anchor_w))
            cv2.circle(overlay, (xx, yy), 3, (0, 0, 255), -1)
        if names is None:
            case_name = f"sample_{start_index:04d}"
        elif isinstance(names, (list, tuple)):
            case_name = str(names[item_index])
        else:
            case_name = str(names)
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in case_name)
        path = os.path.join(visual_dir, f"{start_index:04d}_{safe_name}.png")
        cv2.imwrite(path, overlay)
        start_index += 1
    return start_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--crop_list", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--max_visuals", type=int, default=4)
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument("--stage_topology_stages", type=str, default="none")
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    parser.add_argument(
        "--structure_profile",
        type=str,
        default=STRUCTURE_PROFILE_FULL,
        choices=[
            STRUCTURE_PROFILE_FULL,
            STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
            STRUCTURE_PROFILE_STAGE23_BOUNDARY_FINAL_SKE,
        ],
    )
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23")
    parser.add_argument("--highres_structure_fusion_mode", type=str, default="stage23")
    parser.add_argument("--enable_post_refine_structure_interaction", action="store_true")
    parser.add_argument("--enable_global_topology", action="store_true")
    parser.add_argument("--global_topology_max_nodes", type=int, default=32)
    parser.add_argument("--global_topology_heads", type=int, default=4)
    parser.add_argument("--global_topology_alpha_max", type=float, default=0.05)
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
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
    parser.add_argument("--n_class", default=2, type=int)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.model_path, map_location="cpu")
    inherit_checkpoint_args(args, checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {args.model_path}")
    print(
        "Ablations: baseline=global off; z_struct=only structure token; "
        "z_struct_connectivity=structure+local connectivity token; "
        "z_struct_direction=structure+direction token; full=current fused token.",
        flush=True,
    )

    models = {}
    global_modules = {}
    for mode in MODES:
        model = build_model(
            args,
            checkpoint,
            enable_global_topology=(mode != "baseline"),
        ).to(device)
        model.eval()
        gt = force_global_topology_mode(model, mode)
        gt.capture_diagnostics = True
        models[mode] = model
        global_modules[mode] = gt
        raw_alpha = float(gt.raw_alpha.detach().cpu())
        alpha = float(gt.alpha_global.detach().cpu())
        print(f"{mode:>13} raw_alpha={raw_alpha:.8f} alpha_global={alpha:.8f}")

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
        crop_list_path=args.crop_list,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"{args.split} set size: {len(dataset)}")

    stats = ShiftStats()
    diag_values = {
        mode: {
            "residual_norm": [],
            "feature_delta": [],
            "surface_gate_mean": [],
            "anchor_count": [],
        }
        for mode in MODES
    }
    visual_counts = {mode: 0 for mode in MODES}
    batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="diagnose"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits_by_mode = {}
            for mode, model in models.items():
                outputs = model(images)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                logits_by_mode[mode] = logits
                target = resize_target(masks, logits.shape[-2:])
                metrics = binary_metrics_from_logits(
                    logits,
                    target,
                    threshold=args.threshold,
                )
                stats.metrics[mode].append(metrics)
                gt = global_modules[mode]
                diag = getattr(gt, "last_diagnostics", None)
                if isinstance(diag, dict):
                    if "global_residual_relative_norm" in diag:
                        diag_values[mode]["residual_norm"].append(
                            float(diag["global_residual_relative_norm"].mean().cpu())
                        )
                    if "feature_delta_abs_mean" in diag:
                        diag_values[mode]["feature_delta"].append(
                            float(diag["feature_delta_abs_mean"].mean().cpu())
                        )
                    if "surface_gate_mean" in diag:
                        diag_values[mode]["surface_gate_mean"].append(
                            float(diag["surface_gate_mean"].mean().cpu())
                        )
                    if "anchor_count" in diag:
                        diag_values[mode]["anchor_count"].append(
                            float(diag["anchor_count"].mean().cpu())
                        )
                    if mode != "baseline" and args.max_visuals > 0:
                        visual_counts[mode] = save_attention_visuals(
                            args.output_dir,
                            mode,
                            batch,
                            diag,
                            visual_counts[mode],
                            args.max_visuals,
                        )
            baseline_logits = logits_by_mode["baseline"]
            for mode in MODES:
                if mode == "baseline":
                    continue
                stats.add_shift(
                    mode,
                    baseline_logits,
                    logits_by_mode[mode],
                    masks,
                    args.threshold,
                )
            batches += 1
            if args.max_batches > 0 and batches >= args.max_batches:
                break

    print("\nMetrics")
    print("mode           IoU      F1       Precision  Recall")
    rows = []
    for mode in MODES:
        row_metrics = {
            name: mean_or_zero([m[name] for m in stats.metrics[mode]])
            for name in ("iou", "f1", "precision", "recall")
        }
        print(
            f"{mode:<14} {row_metrics['iou']:.4f}   {row_metrics['f1']:.4f}   "
            f"{row_metrics['precision']:.4f}     {row_metrics['recall']:.4f}"
        )
        rows.append(
            [
                mode,
                row_metrics["iou"],
                row_metrics["f1"],
                row_metrics["precision"],
                row_metrics["recall"],
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )

    print("\nProbability Shift vs Baseline")
    print("mode           region  mean_abs_delta_logits")
    for mode in MODES:
        if mode == "baseline":
            continue
        for region in ("all", "TP", "FN", "FP", "TN"):
            count = stats.delta_count[mode][region]
            mean_delta = stats.delta_sum[mode][region] / max(count, 1)
            print(f"{mode:<14} {region:<6} {mean_delta:.8f}")
            rows.append([mode, "", "", "", "", region, mean_delta, "", "", "", "", ""])

    print("\nPixel Transitions vs Baseline")
    print("mode           TP_to_FN  FN_to_TP  FP_to_TN  TN_to_FP  changed  Q")
    for mode, tr in stats.transitions.items():
        good = tr["FN_to_TP"] + tr["FP_to_TN"]
        q = good / max(tr["changed"], 1)
        print(
            f"{mode:<14} {tr['TP_to_FN']:<8d} {tr['FN_to_TP']:<8d} "
            f"{tr['FP_to_TN']:<8d} {tr['TN_to_FP']:<8d} {tr['changed']:<8d} {q:.4f}"
        )
        rows.append(
            [
                mode,
                "",
                "",
                "",
                "",
                "",
                "",
                tr["TP_to_FN"],
                tr["FN_to_TP"],
                tr["FP_to_TN"],
                tr["TN_to_FP"],
                q,
            ]
        )

    print("\nGlobal Topology Diagnostics")
    print("mode                  raw_alpha   alpha_global  residual_norm  feature_delta  surface_gate_mean  anchor_count")
    for mode, gt in global_modules.items():
        raw_alpha = float(gt.raw_alpha.detach().cpu())
        alpha = float(gt.alpha_global.detach().cpu())
        residual_norm = mean_or_zero(diag_values[mode]["residual_norm"])
        feature_delta = mean_or_zero(diag_values[mode]["feature_delta"])
        surface_gate_mean = mean_or_zero(diag_values[mode]["surface_gate_mean"])
        anchor_count = mean_or_zero(diag_values[mode]["anchor_count"])
        print(
            f"{mode:<21} {raw_alpha:.8f}  {alpha:.8f}  {residual_norm:.8f}  "
            f"{feature_delta:.8f}  {surface_gate_mean:.8f}  {anchor_count:.2f}"
        )
    if args.output_dir:
        print(f"\nSaved attention maps under: {os.path.join(args.output_dir, 'attention_maps')}")

    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "mode",
                    "iou",
                    "f1",
                    "precision",
                    "recall",
                    "region",
                    "mean_abs_delta_logits",
                    "TP_to_FN",
                    "FN_to_TP",
                    "FP_to_TN",
                    "TN_to_FP",
                    "Q",
                ]
            )
            writer.writerows(rows)
        print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
