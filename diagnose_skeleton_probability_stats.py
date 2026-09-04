import argparse
import csv
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.vision_transformer import load_topology_checkpoint_state


class ConfigArgs:
    root_path = "./data1"
    dataset = "ImageData"
    list_dir = "./lists/lists_Synapse"
    num_classes = 2
    cfg = "./configs/swin_tiny_patch4_window7_224_lite.yaml"
    img_size = 256
    batch_size = 1
    num_workers = 0
    zip = False
    cache_mode = ""
    resume = ""
    accumulation_steps = 0
    use_checkpoint = False
    amp_opt_level = ""
    tag = ""
    eval = True
    throughput = False
    n_class = 2
    opts = None


def update_args_from_checkpoint(args, checkpoint):
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if isinstance(saved_args, dict):
        for name, value in saved_args.items():
            setattr(args, name, value)
    return args


def build_model(args, checkpoint, device):
    config = get_config(args)
    model = ViT_seg(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        use_asterisk=True,
        return_skeleton=True,
        bottleneck_type=getattr(args, "bottleneck_type", "global_local"),
        structure_profile=getattr(args, "structure_profile", "full"),
        use_msfe_skip=not getattr(args, "disable_msfe_skip", False),
        enable_highres_structure_stream=getattr(args, "enable_highres_structure_stream", False),
        highres_structure_channels=getattr(args, "highres_structure_channels", 64),
        highres_structure_fuse_stages=getattr(args, "highres_structure_fuse_stages", "stage23"),
        highres_structure_fusion_mode=getattr(args, "highres_structure_fusion_mode", "stage23"),
        enable_post_refine_structure_interaction=getattr(
            args,
            "enable_post_refine_structure_interaction",
            False,
        ),
        enable_global_topology=getattr(args, "enable_global_topology", True),
        global_topology_max_nodes=getattr(args, "global_topology_max_nodes", 32),
        global_topology_heads=getattr(args, "global_topology_heads", 4),
        global_topology_reach_hops=getattr(args, "global_topology_reach_hops", 12),
        global_topology_nms_radius=getattr(args, "global_topology_nms_radius", 2),
        global_topology_skeleton_threshold=getattr(args, "global_topology_skeleton_threshold", 0.5),
        global_topology_connectivity_threshold=getattr(args, "global_topology_connectivity_threshold", 0.25),
        global_topology_bend_angle_threshold=getattr(args, "global_topology_bend_angle_threshold", 45.0),
        global_topology_alpha_max=getattr(args, "global_topology_alpha_max", 0.05),
    )
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=True,
    )
    return model.to(device).eval()


def find_stage3_output(stage_outputs):
    for item in reversed(stage_outputs):
        if item.get("stage") == 3:
            return item
    return None


def find_highres_structure_skeleton(stage_outputs):
    for item in reversed(stage_outputs):
        if item.get("stage") == "highres_structure":
            skeleton = item.get("highres_structure_skeleton")
            if skeleton is not None:
                return skeleton
    return None


def resolve_skeleton_probability(outputs, target_size=None):
    stage_outputs = outputs[4] if len(outputs) > 4 else []
    stage3 = find_stage3_output(stage_outputs)
    if stage3 is not None and stage3.get("skeleton") is not None:
        prob = torch.sigmoid(stage3["skeleton"])
        return prob, "stage3_skeleton"

    highres_skeleton = find_highres_structure_skeleton(stage_outputs)
    if highres_skeleton is not None:
        prob = torch.sigmoid(highres_skeleton)
        source = "highres_structure_skeleton"
    elif len(outputs) > 2 and outputs[2] is not None:
        prob = torch.sigmoid(outputs[2])
        source = "final_skeleton"
    else:
        raise RuntimeError("No skeleton logits found in model outputs.")

    if target_size is not None and prob.shape[-2:] != target_size:
        prob = F.interpolate(prob, size=target_size, mode="bilinear", align_corners=False)
    return prob, source


def tensor_stats(values):
    values = values.detach().float().flatten().cpu()
    percentiles = torch.quantile(
        values,
        torch.tensor([0.50, 0.75, 0.90, 0.95, 0.99], dtype=torch.float32),
    )
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "p50": float(percentiles[0].item()),
        "p75": float(percentiles[1].item()),
        "p90": float(percentiles[2].item()),
        "p95": float(percentiles[3].item()),
        "p99": float(percentiles[4].item()),
        "pixels": int(values.numel()),
        "count_gt_005": int((values > 0.05).sum().item()),
        "count_gt_010": int((values > 0.10).sum().item()),
        "count_gt_020": int((values > 0.20).sum().item()),
        "count_gt_050": int((values > 0.50).sum().item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Print skeleton probability distribution statistics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--cfg", default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument(
        "--target_resolution",
        choices=("native", "input"),
        default="native",
        help="Report native skeleton output resolution or upsample to input resolution.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[skel-prob] device={device}", flush=True)
    print(f"[skel-prob] loading checkpoint: {args.checkpoint}", flush=True)
    start_time = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"[skel-prob] checkpoint loaded in {time.perf_counter() - start_time:.2f}s", flush=True)

    config_args = update_args_from_checkpoint(ConfigArgs(), checkpoint)
    config_args.root_path = args.root_path
    config_args.img_size = args.img_size
    config_args.batch_size = args.batch_size
    config_args.num_workers = args.num_workers
    config_args.cfg = args.cfg

    print("[skel-prob] building model", flush=True)
    start_time = time.perf_counter()
    model = build_model(config_args, checkpoint, device)
    print(f"[skel-prob] model ready in {time.perf_counter() - start_time:.2f}s", flush=True)

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"[skel-prob] dataset size={len(dataset)}", flush=True)

    rows = []
    all_values = []
    source_counts = {}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            should_print = args.print_freq > 0 and batch_index % args.print_freq == 0
            if should_print:
                print(f"[skel-prob] batch {batch_index + 1} forward start", flush=True)
            images = batch["image"].to(device)
            outputs = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            target_size = images.shape[-2:] if args.target_resolution == "input" else None
            skeleton_prob, source = resolve_skeleton_probability(outputs, target_size=target_size)
            source_counts[source] = source_counts.get(source, 0) + int(skeleton_prob.shape[0])
            stats = tensor_stats(skeleton_prob)
            stats["batch_index"] = batch_index
            stats["source"] = source
            stats["height"] = int(skeleton_prob.shape[-2])
            stats["width"] = int(skeleton_prob.shape[-1])
            rows.append(stats)
            all_values.append(skeleton_prob.detach().float().flatten().cpu())
            if should_print:
                print(
                    "[skel-prob] batch {} done: source={} shape={}x{} mean={:.6f} p95={:.6f} >0.05={} >0.5={}".format(
                        batch_index + 1,
                        source,
                        stats["height"],
                        stats["width"],
                        stats["mean"],
                        stats["p95"],
                        stats["count_gt_005"],
                        stats["count_gt_050"],
                    ),
                    flush=True,
                )

    if not rows:
        raise RuntimeError("No batches were processed.")

    all_values = torch.cat(all_values, dim=0)
    summary = tensor_stats(all_values)
    summary["split"] = args.split
    summary["batches"] = len(rows)
    summary["target_resolution"] = args.target_resolution
    summary["sources"] = ";".join(f"{key}:{value}" for key, value in sorted(source_counts.items()))

    cases_path = os.path.join(args.output_dir, "skeleton_probability_cases.csv")
    with open(cases_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.join(args.output_dir, "skeleton_probability_summary.csv")
    with open(summary_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())

    print("\nSkeleton probability statistics", flush=True)
    for key in (
        "mean",
        "std",
        "min",
        "max",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "pixels",
        "count_gt_005",
        "count_gt_010",
        "count_gt_020",
        "count_gt_050",
        "sources",
    ):
        print(f"{key}={summary[key]}", flush=True)
    print("saved:", summary_path, flush=True)
    print("saved:", cases_path, flush=True)


if __name__ == "__main__":
    main()
