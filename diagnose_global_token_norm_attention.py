from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_structure_supervision import adapt_connectivity_modules_for_checkpoint
from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import (
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_FINAL_SKE,
    SwinUnet,
    load_topology_checkpoint_state,
)


def cli_has(name):
    return name in sys.argv[1:] or name.replace("_", "-") in sys.argv[1:]


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
        "stage_topology_bias_mode",
        "stage_topology_ratio",
        "stage_topology_topo_clip",
        "stage2_skeleton_gradient_ratio",
        "stage3_skeleton_gradient_ratio",
        "final_skeleton_gradient_ratio",
        "enable_highres_structure_stream",
        "highres_structure_channels",
        "highres_structure_fuse_stages",
        "highres_structure_fusion_mode",
        "enable_post_refine_structure_interaction",
        "enable_global_topology",
        "global_topology_max_nodes",
        "global_topology_heads",
        "global_topology_alpha_max",
    ):
        if name in saved_args and not cli_has("--" + name):
            setattr(args, name, saved_args[name])
    if checkpoint.get("structure_profile") and not cli_has("--structure_profile"):
        args.structure_profile = checkpoint["structure_profile"]


def build_model(args, checkpoint, device):
    config = get_config(args)
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
        enable_post_refine_structure_interaction=args.enable_post_refine_structure_interaction,
        enable_global_topology=args.enable_global_topology,
        global_topology_max_nodes=args.global_topology_max_nodes,
        global_topology_heads=args.global_topology_heads,
        global_topology_alpha_max=args.global_topology_alpha_max,
    )
    model_state = model.state_dict()
    state_dict = {}
    skipped = []
    for key, value in checkpoint["model_state_dict"].items():
        if (
            key.startswith("swin_unet.global_topology.")
            and key in model_state
            and hasattr(value, "shape")
            and value.shape != model_state[key].shape
        ):
            skipped.append(key)
            continue
        state_dict[key] = value
    if skipped:
        print(
            "[WARN] Reinitialized shape-incompatible global_topology keys: "
            + ", ".join(skipped),
            flush=True,
        )
    adapt_connectivity_modules_for_checkpoint(model, state_dict, "standard")
    load_topology_checkpoint_state(
        model,
        state_dict,
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=False,
    )
    return model.to(device).eval()


def tensor_norm_mean(x, valid):
    values = torch.linalg.vector_norm(x.detach().float(), dim=-1)
    values = values[valid]
    if values.numel() == 0:
        return 0.0
    return float(values.mean().cpu())


def cosine_mean(a, b, valid):
    value = F.cosine_similarity(a.detach().float(), b.detach().float(), dim=-1)
    value = value[valid]
    if value.numel() == 0:
        return float("nan")
    return float(value.mean().cpu())


def corrcoef(x, y):
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    ok = torch.isfinite(x) & torch.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom.item() <= 0:
        return float("nan")
    return float((x * y).sum().div(denom).cpu())


def analyze_global_call(module, feature, z_struct, surface_prob, connectivity_feature, direction_feature):
    batch, channels, height, width = feature.shape
    with torch.no_grad():
        z_score = torch.linalg.vector_norm(z_struct.detach().float(), dim=1, keepdim=True)
        z_score = module._minmax_normalize_map(z_score).to(dtype=feature.dtype)
        if z_score.shape[-2:] != (height, width):
            z_score = F.interpolate(z_score, size=(height, width), mode="bilinear", align_corners=False)
        surface_gate = surface_prob.detach().to(dtype=feature.dtype)
        if surface_gate.shape[-2:] != (height, width):
            surface_gate = F.interpolate(surface_gate, size=(height, width), mode="bilinear", align_corners=False)
        anchor_score = z_score * surface_gate.clamp(0.0, 1.0)
        coords, valid, scores, candidate_count = module._extract_fps_anchors(anchor_score)

        sampled_struct = module._sample_features_at_anchor_coords(
            z_struct.to(dtype=feature.dtype), coords, anchor_hw=(height, width)
        )
        sampled_feature = module._sample_features_at_anchor_coords(
            feature, coords, anchor_hw=(height, width)
        )
        connectivity_map = module._prepare_token_map(
            connectivity_feature,
            batch,
            (height, width),
            module.connectivity_channels,
            feature.dtype,
            feature.device,
        )
        direction_map = module._prepare_token_map(
            direction_feature,
            batch,
            (height, width),
            module.direction_channels,
            feature.dtype,
            feature.device,
        )
        sampled_connectivity = module._sample_features_at_anchor_coords(
            connectivity_map, coords, anchor_hw=(height, width)
        )
        sampled_direction = module._sample_features_at_anchor_coords(
            direction_map, coords, anchor_hw=(height, width)
        )
        node_types = torch.zeros(batch, module.max_nodes, device=feature.device, dtype=torch.long)
        coords_norm = coords.float() / feature.new_tensor([max(height - 1, 1), max(width - 1, 1)])
        meta_input = torch.cat([module.node_type_embedding(node_types), coords_norm], dim=-1)

        struct_proj = module.struct_token_projection(sampled_struct)
        decoder_proj = module.decoder_token_projection(sampled_feature)
        conn_proj = module.connectivity_token_projection(sampled_connectivity)
        dir_proj = module.direction_token_projection(sampled_direction)
        meta_proj = module.meta_token_projection(meta_input)
        node_feature = struct_proj + decoder_proj + conn_proj + dir_proj + meta_proj

        qkv = module.token_relation_qkv(node_feature).reshape(
            batch,
            module.max_nodes,
            3,
            module.heads,
            channels // module.heads,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        relation_logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(channels // module.heads)
        topology_bias = module._relative_topology_bias(coords, valid, (height, width))
        relation_logits = relation_logits + module.token_relation_scale * topology_bias
        relation_logits = relation_logits.masked_fill(
            ~valid[:, None, None, :], -torch.finfo(relation_logits.dtype).max
        )
        relation_attention = torch.softmax(relation_logits, dim=-1)

        refined = torch.matmul(relation_attention, value).transpose(1, 2).reshape(
            batch, module.max_nodes, channels
        )
        refined = module.token_relation_projection(refined)
        refined_node = (node_feature + refined) * valid.unsqueeze(-1).to(dtype=node_feature.dtype)

        grid_tokens = feature.flatten(2).transpose(1, 2)
        grid_query = module.grid_q(grid_tokens).reshape(
            batch, height * width, module.heads, channels // module.heads
        ).permute(0, 2, 1, 3)
        node_kv = module.node_kv(refined_node).reshape(
            batch, module.max_nodes, 2, module.heads, channels // module.heads
        ).permute(2, 0, 3, 1, 4)
        node_key = node_kv[0]
        cross_logits = torch.matmul(grid_query, node_key.transpose(-2, -1)) / math.sqrt(channels // module.heads)
        cross_logits = cross_logits.masked_fill(
            ~valid[:, None, None, :], -torch.finfo(cross_logits.dtype).max
        )
        cross_attention = torch.softmax(cross_logits, dim=-1)

        conn_anchor = F.normalize(sampled_connectivity.detach().float(), dim=-1, eps=1e-6)
        conn_grid = F.normalize(
            connectivity_map.detach().float().flatten(2).transpose(1, 2),
            dim=-1,
            eps=1e-6,
        )
        dir_anchor = F.normalize(sampled_direction.detach().float(), dim=-1, eps=1e-6)
        dir_grid = F.normalize(
            direction_map.detach().float().flatten(2).transpose(1, 2),
            dim=-1,
            eps=1e-6,
        )
        relation_conn_similarity = torch.matmul(conn_anchor, conn_anchor.transpose(1, 2))
        relation_dir_similarity = torch.matmul(dir_anchor, dir_anchor.transpose(1, 2))
        relation_distance = torch.sqrt(
            (
                coords_norm[:, :, None, :]
                - coords_norm[:, None, :, :]
            ).square().sum(dim=-1)
            + 1e-8
        )
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, device=feature.device),
            torch.linspace(0.0, 1.0, width, device=feature.device),
            indexing="ij",
        )
        grid_coords = torch.stack([grid_y, grid_x], dim=-1).reshape(
            1,
            height * width,
            1,
            2,
        )
        cross_conn_similarity = torch.matmul(conn_grid, conn_anchor.transpose(1, 2))
        cross_dir_similarity = torch.matmul(dir_grid, dir_anchor.transpose(1, 2))
        cross_distance = torch.sqrt(
            (
                grid_coords
                - coords_norm[:, None, :, :]
            ).square().sum(dim=-1)
            + 1e-8
        )
        relation_mask = valid[:, :, None] & valid[:, None, :]
        cross_mask = valid[:, None, :].expand(batch, height * width, module.max_nodes)
        head_rows = []
        for head in range(module.heads):
            head_rows.append(
                {
                    "head": head,
                    "relation_conn_corr": corrcoef(
                        relation_attention[:, head][relation_mask],
                        relation_conn_similarity[relation_mask],
                    ),
                    "relation_dir_corr": corrcoef(
                        relation_attention[:, head][relation_mask],
                        relation_dir_similarity[relation_mask],
                    ),
                    "relation_distance_corr": corrcoef(
                        relation_attention[:, head][relation_mask],
                        relation_distance[relation_mask],
                    ),
                    "cross_conn_corr": corrcoef(
                        cross_attention[:, head][cross_mask],
                        cross_conn_similarity[cross_mask],
                    ),
                    "cross_dir_corr": corrcoef(
                        cross_attention[:, head][cross_mask],
                        cross_dir_similarity[cross_mask],
                    ),
                    "cross_distance_corr": corrcoef(
                        cross_attention[:, head][cross_mask],
                        cross_distance[cross_mask],
                    ),
                }
            )

        return {
            "anchor_count": float(valid.sum().cpu()),
            "candidate_count": float(candidate_count),
            "alpha_global": float(module.alpha_global.detach().cpu()),
            "token_relation_scale": float(module.token_relation_scale.detach().cpu()),
            "pre_z_struct_norm": tensor_norm_mean(sampled_struct, valid),
            "pre_decoder_norm": tensor_norm_mean(sampled_feature, valid),
            "pre_conn_norm": tensor_norm_mean(sampled_connectivity, valid),
            "pre_dir_norm": tensor_norm_mean(sampled_direction, valid),
            "post_z_struct_norm": tensor_norm_mean(struct_proj, valid),
            "post_decoder_norm": tensor_norm_mean(decoder_proj, valid),
            "post_conn_norm": tensor_norm_mean(conn_proj, valid),
            "post_dir_norm": tensor_norm_mean(dir_proj, valid),
            "post_meta_norm": tensor_norm_mean(meta_proj, valid),
            "cos_conn_decoder": cosine_mean(conn_proj, decoder_proj, valid),
            "cos_conn_z_struct": cosine_mean(conn_proj, struct_proj, valid),
            "cos_dir_decoder": cosine_mean(dir_proj, decoder_proj, valid),
            "cos_dir_conn": cosine_mean(dir_proj, conn_proj, valid),
            "head_rows": head_rows,
        }


def mean_or_nan(values):
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=5)
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--cfg", type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--dataset", type=str, default="ImageData")
    parser.add_argument("--n_class", default=2, type=int)
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
    parser.add_argument("--structure_profile", type=str, default=STRUCTURE_PROFILE_STAGE23_BOUNDARY_FINAL_SKE)
    parser.add_argument("--bottleneck_type", type=str, default="global_local")
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
    parser.add_argument("--enable_post_refine_structure_interaction", action="store_true")
    parser.add_argument("--enable_global_topology", action="store_true")
    parser.add_argument("--global_topology_max_nodes", type=int, default=32)
    parser.add_argument("--global_topology_heads", type=int, default=4)
    parser.add_argument("--global_topology_alpha_max", type=float, default=0.05)
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    inherit_checkpoint_args(args, checkpoint)
    model = build_model(args, checkpoint, device)
    swin_unet = getattr(model, "swin_unet", model)
    global_topology = getattr(swin_unet, "global_topology", None)
    if global_topology is None:
        raise RuntimeError("model.swin_unet.global_topology was not found")
    if not getattr(global_topology, "enable_global_topology", False):
        raise RuntimeError("global topology is disabled; pass --enable_global_topology or use a checkpoint that saved it")

    rows = []
    captured_calls = []
    original_forward = global_topology.forward_feature_anchors

    def wrapped_forward(self, feature, z_struct, surface_prob, connectivity_feature=None, direction_feature=None):
        if len(captured_calls) < args.max_batches:
            captured_calls.append(
                analyze_global_call(
                    self,
                    feature,
                    z_struct,
                    surface_prob,
                    connectivity_feature,
                    direction_feature,
                )
            )
        return original_forward(feature, z_struct, surface_prob, connectivity_feature, direction_feature)

    global_topology.forward_feature_anchors = MethodType(wrapped_forward, global_topology)

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
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= args.max_batches:
                break
            image = batch["image"].to(device, non_blocking=True)
            skeleton = batch.get("skeleton")
            skeleton = skeleton.to(device, non_blocking=True) if skeleton is not None else None
            model(image, gt_skeleton=skeleton)

    global_topology.forward_feature_anchors = original_forward
    rows = captured_calls
    if not rows:
        raise RuntimeError("No global topology calls were captured")

    scalar_keys = [key for key in rows[0] if key != "head_rows"]
    print("\nGlobal Token Norm Diagnostic")
    print("============================")
    print(f"checkpoint: {args.model_path}")
    print(f"split={args.split} batches={len(rows)} batch_size={args.batch_size}")
    for key in scalar_keys:
        print(f"{key:<28} {mean_or_nan([row[key] for row in rows]):.8f}")

    head_count = len(rows[0]["head_rows"])
    print("\nPer-head attention correlation")
    print("head  rel_conn  rel_dir  rel_dist  cross_conn  cross_dir  cross_dist")
    head_output_rows = []
    for head in range(head_count):
        out = {
            "head": head,
            "relation_conn_corr": mean_or_nan(
                [row["head_rows"][head]["relation_conn_corr"] for row in rows]
            ),
            "relation_dir_corr": mean_or_nan(
                [row["head_rows"][head]["relation_dir_corr"] for row in rows]
            ),
            "relation_distance_corr": mean_or_nan(
                [row["head_rows"][head]["relation_distance_corr"] for row in rows]
            ),
            "cross_conn_corr": mean_or_nan(
                [row["head_rows"][head]["cross_conn_corr"] for row in rows]
            ),
            "cross_dir_corr": mean_or_nan(
                [row["head_rows"][head]["cross_dir_corr"] for row in rows]
            ),
            "cross_distance_corr": mean_or_nan(
                [row["head_rows"][head]["cross_distance_corr"] for row in rows]
            ),
        }
        head_output_rows.append(out)
        print(
            f"{head:<4d} "
            f"{out['relation_conn_corr']:>8.6f} "
            f"{out['relation_dir_corr']:>7.6f} "
            f"{out['relation_distance_corr']:>8.6f} "
            f"{out['cross_conn_corr']:>10.6f} "
            f"{out['cross_dir_corr']:>9.6f} "
            f"{out['cross_distance_corr']:>10.6f}"
        )

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=scalar_keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in scalar_keys})
        head_csv = os.path.splitext(args.output_csv)[0] + "_heads.csv"
        with open(head_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(head_output_rows[0].keys()))
            writer.writeheader()
            writer.writerows(head_output_rows)
        print(f"\nSaved CSV: {args.output_csv}")
        print(f"Saved head CSV: {head_csv}")


if __name__ == "__main__":
    main()
