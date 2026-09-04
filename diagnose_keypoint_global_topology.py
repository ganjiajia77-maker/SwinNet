import argparse
import csv
import os
import sys
import types

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.keypoint_global_topology import KeypointGuidedGlobalTopology
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.vision_transformer import load_topology_checkpoint_state


CONNECTIVITY_DIRECTIONS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)


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


def make_topology_module(max_nodes=16):
    return KeypointGuidedGlobalTopology(
        channels=8,
        max_nodes=max_nodes,
        heads=2,
        reach_hops=16,
        skeleton_threshold=0.1,
        connectivity_threshold=0.2,
        bend_angle_threshold=45.0,
        enabled=True,
    )


def c8_grid(height=11, width=11, fill=0.01):
    return torch.full((1, 8, height, width), fill)


def connect(c8, y, x, dy, dx, value=0.9):
    index = CONNECTIVITY_DIRECTIONS.index((dy, dx))
    c8[0, index, y, x] = value


def connect_bidirectional(c8, y, x, dy, dx, value=0.9):
    connect(c8, y, x, dy, dx, value)
    connect(c8, y + dy, x + dx, -dy, -dx, value)


def mark_horizontal(c8, y, xs, value=0.9):
    for x in xs:
        connect(c8, y, x, 0, 1, value)
        connect(c8, y, x, 0, -1, value)


def mark_vertical(c8, ys, x, value=0.9):
    for y in ys:
        connect(c8, y, x, 1, 0, value)
        connect(c8, y, x, -1, 0, value)


def reach_from_seed(module, connectivity, seed_points):
    symmetric = module.build_symmetric_connectivity(connectivity)
    seeds = torch.zeros(1, len(seed_points), *connectivity.shape[-2:])
    for index, (y, x) in enumerate(seed_points):
        seeds[0, index, y, x] = 1
    return module.build_multisource_reachability(seeds, symmetric)


def run_synthetic(args):
    module = make_topology_module()
    rows = []

    c8 = c8_grid()
    mark_horizontal(c8, 5, range(2, 9))
    reach = reach_from_seed(module, c8, [(5, 2), (5, 8)])
    rows.append(("case1_straight", "Reach(A,B)", float(reach[0, 0, 5, 8]), "high"))

    c8 = c8_grid()
    mark_horizontal(c8, 4, range(2, 7))
    mark_vertical(c8, range(4, 9), 6)
    mark_horizontal(c8, 8, range(6, 10))
    reach = reach_from_seed(module, c8, [(4, 2), (8, 9)])
    rows.append(("case2_bend", "Reach(A,D)", float(reach[0, 0, 8, 9]), ">0"))
    rows.append(("case2_bend", "straight_chord_leak", float(reach[0, 0, 6, 5]), "low"))

    c8 = c8_grid()
    mark_horizontal(c8, 4, range(2, 10))
    mark_horizontal(c8, 6, range(2, 10))
    reach = reach_from_seed(module, c8, [(4, 5), (6, 5)])
    rows.append(("case3_parallel", "Reach(A,C)", float(reach[0, 0, 6, 5]), "low"))

    c8 = c8_grid()
    mark_horizontal(c8, 5, range(2, 9))
    mark_vertical(c8, range(2, 9), 5)
    reach = reach_from_seed(module, c8, [(5, 2), (5, 8), (2, 5), (8, 5)])
    rows.append(("case4_intersection", "Reach(A,B)", float(reach[0, 0, 5, 8]), "high"))
    rows.append(("case4_intersection", "Reach(A,C)", float(reach[0, 0, 2, 5]), "high"))
    rows.append(("case4_intersection", "Reach(A,D)", float(reach[0, 0, 8, 5]), "high"))

    c8 = c8_grid()
    mark_horizontal(c8, 5, range(2, 5))
    mark_horizontal(c8, 5, range(7, 10))
    reach = reach_from_seed(module, c8, [(5, 2), (5, 9)])
    rows.append(("case5_broken_middle", "Reach(A,B)", float(reach[0, 0, 5, 9]), "near_zero"))

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "synthetic_5cases.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "metric", "value", "expected"])
        writer.writerows(rows)

    print("case,metric,value,expected")
    for row in rows:
        print("{},{},{:.6f},{}".format(row[0], row[1], row[2], row[3]))
    print("saved:", path)

    failures = []
    for case, metric, value, expected in rows:
        if expected == "high" and value <= 0.5:
            failures.append((case, metric, value, expected))
        elif expected == ">0" and value <= 0.01:
            failures.append((case, metric, value, expected))
        elif expected in ("low", "near_zero") and value >= 0.01:
            failures.append((case, metric, value, expected))
    if failures:
        raise SystemExit("synthetic failures: {}".format(failures))


def patch_global_topology_capture(module):
    original_forward = module.forward

    def wrapped(self, feature, skeleton_prob, connectivity_prob, direction):
        if not self.enable_global_topology:
            return original_forward(feature, skeleton_prob, connectivity_prob, direction)

        batch, channels, height, width = feature.shape
        with torch.no_grad():
            symmetric = self.build_symmetric_connectivity(connectivity_prob.detach())
            coords, node_types, valid, scores = self.extract_keypoints(
                skeleton_prob.detach(),
                symmetric,
                direction.detach(),
            )
            seeds = feature.new_zeros((batch, self.max_nodes, height, width))
            seed_flat = seeds.flatten(2)
            seed_index = (coords[..., 0] * width + coords[..., 1]).unsqueeze(-1)
            seed_flat.scatter_(2, seed_index, valid.unsqueeze(-1).float())
            reach = self.build_multisource_reachability(seed_flat.view_as(seeds), symmetric)
            adjacency = self.build_node_adjacency(reach, coords, valid, direction.detach())
            g_global = (reach * skeleton_prob.detach()).amax(dim=1, keepdim=True)

        self.last_reach = reach.detach().cpu()
        self.last_g_global = g_global.detach().cpu()
        self.last_coords = coords.detach().cpu()
        self.last_valid = valid.detach().cpu()
        self.last_adjacency = adjacency.detach().cpu()
        return original_forward(feature, skeleton_prob, connectivity_prob, direction)

    module.forward = types.MethodType(wrapped, module)


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
        enable_global_topology=True,
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


def coverage(mask, g_global, threshold):
    total = int(mask.sum().item())
    if total == 0:
        return float("nan"), total, 0
    hit = int(((g_global > threshold) & mask).sum().item())
    return hit / total, total, hit


def run_coverage(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_args = update_args_from_checkpoint(ConfigArgs(), checkpoint)
    config_args.root_path = args.root_path
    config_args.img_size = args.img_size
    config_args.batch_size = args.batch_size
    config_args.num_workers = args.num_workers
    config_args.cfg = args.cfg

    model = build_model(config_args, checkpoint, device)
    global_topology = model.swin_unet.global_topology
    global_topology.capture_diagnostics = True
    patch_global_topology_capture(global_topology)

    alpha = float(global_topology.alpha_global.detach().cpu())
    raw_alpha = float(global_topology.raw_alpha.detach().cpu())
    print("raw_alpha={:.8f} alpha_global={:.8f}".format(raw_alpha, alpha))

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

    totals = {
        "tp_pixels": 0,
        "tp_hit": 0,
        "fp_pixels": 0,
        "fp_hit": 0,
        "fn_pixels": 0,
        "fn_hit": 0,
    }
    attention_same_values = []
    attention_different_values = []
    attention_delta_values = []
    rows = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            if masks.shape[1] != 1:
                masks = masks[:, :1]
            gt_mask = masks > 0.5

            old_enabled = global_topology.enable_global_topology
            global_topology.enable_global_topology = False
            baseline_logits = model(images)[0]
            baseline_pred = torch.sigmoid(baseline_logits) > args.pred_threshold

            global_topology.enable_global_topology = True
            global_logits = model(images)[0]
            global_topology.enable_global_topology = old_enabled

            g_global = global_topology.last_g_global.to(device)
            if g_global.shape[-2:] != gt_mask.shape[-2:]:
                g_global = F.interpolate(
                    g_global,
                    size=gt_mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            tp = gt_mask & baseline_pred
            fp = (~gt_mask) & baseline_pred
            fn = gt_mask & (~baseline_pred)
            tp_cov, tp_total, tp_hit = coverage(tp, g_global, args.global_threshold)
            fp_cov, fp_total, fp_hit = coverage(fp, g_global, args.global_threshold)
            fn_cov, fn_total, fn_hit = coverage(fn, g_global, args.global_threshold)

            totals["tp_pixels"] += tp_total
            totals["tp_hit"] += tp_hit
            totals["fp_pixels"] += fp_total
            totals["fp_hit"] += fp_hit
            totals["fn_pixels"] += fn_total
            totals["fn_hit"] += fn_hit

            delta_prob = torch.sigmoid(global_logits) - torch.sigmoid(baseline_logits)
            fn_delta = float(delta_prob[fn].mean().item()) if fn.any() else float("nan")
            diagnostics = global_topology.last_diagnostics or {}
            attention_same = diagnostics.get("attention_same_mean")
            attention_different = diagnostics.get("attention_different_mean")
            attention_delta = diagnostics.get("attention_delta")
            attention_same = float(torch.nanmean(attention_same).item()) if attention_same is not None else float("nan")
            attention_different = float(torch.nanmean(attention_different).item()) if attention_different is not None else float("nan")
            attention_delta = float(torch.nanmean(attention_delta).item()) if attention_delta is not None else float("nan")
            if not torch.isnan(torch.tensor(attention_same)):
                attention_same_values.append(attention_same)
            if not torch.isnan(torch.tensor(attention_different)):
                attention_different_values.append(attention_different)
            if not torch.isnan(torch.tensor(attention_delta)):
                attention_delta_values.append(attention_delta)

            rows.append(
                [
                    batch_index,
                    tp_total,
                    fp_total,
                    fn_total,
                    tp_cov,
                    fp_cov,
                    fn_cov,
                    fn_delta,
                    attention_same,
                    attention_different,
                    attention_delta,
                    int(global_topology.last_valid.sum().item()),
                    float(global_topology.last_adjacency.mean().item()),
                    float(global_topology.last_adjacency.max().item()),
                ]
            )

    def total_rate(hit_name, total_name):
        total = totals[total_name]
        return float("nan") if total == 0 else totals[hit_name] / total

    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "pred_threshold": args.pred_threshold,
        "global_threshold": args.global_threshold,
        "raw_alpha": raw_alpha,
        "alpha_global": alpha,
        "attention_same_mean": sum(attention_same_values) / max(len(attention_same_values), 1),
        "attention_different_mean": sum(attention_different_values) / max(len(attention_different_values), 1),
        "attention_delta": sum(attention_delta_values) / max(len(attention_delta_values), 1),
        "tp_coverage": total_rate("tp_hit", "tp_pixels"),
        "fp_coverage": total_rate("fp_hit", "fp_pixels"),
        "fn_coverage": total_rate("fn_hit", "fn_pixels"),
        **totals,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "global_fn_coverage_summary.csv")
    with open(summary_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())

    cases_path = os.path.join(args.output_dir, "global_fn_coverage_cases.csv")
    with open(cases_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "batch_index",
                "tp_pixels",
                "fp_pixels",
                "fn_pixels",
                "tp_coverage",
                "fp_coverage",
                "fn_coverage",
                "fn_delta_prob_mean",
                "attention_same_mean",
                "attention_different_mean",
                "attention_delta",
                "node_count",
                "adjacency_mean",
                "adjacency_max",
            ]
        )
        writer.writerows(rows)

    for key, value in summary.items():
        print("{}={}".format(key, value))
    print("saved:", summary_path)
    print("saved:", cases_path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Diagnose keypoint-guided global topology on synthetic cases and FN coverage."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--output_dir", default="./analysis_out/keypoint_global_topology_diag")
    synthetic.set_defaults(func=run_synthetic)

    coverage_parser = subparsers.add_parser("coverage")
    coverage_parser.add_argument("--checkpoint", required=True)
    coverage_parser.add_argument("--root_path", default="./data1")
    coverage_parser.add_argument("--split", default="test")
    coverage_parser.add_argument("--output_dir", default="./analysis_out/keypoint_global_topology_diag")
    coverage_parser.add_argument("--cfg", default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    coverage_parser.add_argument("--img_size", type=int, default=256)
    coverage_parser.add_argument("--batch_size", type=int, default=1)
    coverage_parser.add_argument("--num_workers", type=int, default=0)
    coverage_parser.add_argument("--pred_threshold", type=float, default=0.5)
    coverage_parser.add_argument("--global_threshold", type=float, default=0.1)
    coverage_parser.add_argument("--max_batches", type=int, default=0)
    coverage_parser.set_defaults(func=run_coverage)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
