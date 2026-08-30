import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_structure_supervision import load_model, resize_like, select_stage_outputs
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import build_connectivity_target, build_stage_skeleton_target
from networks.skeleton_guided_head import PairwiseConnectivityHead
from topology_direction_constants import (
    CONNECTIVITY_DIR_NAMES,
    CONNECTIVITY_DIRECTIONS,
    axial_double_angle_basis,
    connectivity_double_angle_basis,
)


def set_deterministic(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def safe_auc(gt, score):
    gt = np.asarray(gt, dtype=np.int32)
    score = np.asarray(score, dtype=np.float32)
    if gt.size == 0 or np.unique(gt).size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(gt, score)), float(average_precision_score(gt, score))


def summarize(name, gt_chunks, score_chunks):
    gt = np.concatenate(gt_chunks, axis=0) if gt_chunks else np.empty((0,), dtype=np.int32)
    score = np.concatenate(score_chunks, axis=0) if score_chunks else np.empty((0,), dtype=np.float32)
    auroc, auprc = safe_auc(gt, score)
    pos = score[gt > 0]
    neg = score[gt == 0]
    return {
        "name": name,
        "auroc": auroc,
        "auprc": auprc,
        "c_pos": float(pos.mean()) if pos.size else float("nan"),
        "c_neg": float(neg.mean()) if neg.size else float("nan"),
        "pos_count": int(pos.size),
        "neg_count": int(neg.size),
    }


def print_row(row):
    print(
        f"{row['name']:24s} "
        f"{row['auroc']:8.4f} "
        f"{row['auprc']:8.4f} "
        f"{row['c_pos']:8.4f} "
        f"{row['c_neg']:8.4f} "
        f"{row['pos_count']:10d} "
        f"{row['neg_count']:10d}"
    )


def find_skeleton_logits(structure_outputs, selected_output, final_skeleton_logits):
    if "skeleton" in selected_output:
        return selected_output["skeleton"], "selected stage_output['skeleton']"

    selected_stage = selected_output.get("stage")
    selected_step = selected_output.get("refinement_step", None)
    fallback = None
    fallback_desc = None
    for item in structure_outputs:
        if item.get("stage") != selected_stage or "skeleton" not in item:
            continue
        if item.get("refinement_step", None) == selected_step:
            return item["skeleton"], "same stage/refinement skeleton"
        fallback = item["skeleton"]
        fallback_desc = "same stage fallback skeleton"

    if fallback is not None:
        return fallback, fallback_desc
    return final_skeleton_logits, "final skeleton logits fallback"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--stage", type=str, default="stage3_refine", choices=["stage2_refine", "stage3_refine"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
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
    parser.add_argument("--disable_msfe_skip", action="store_true")
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
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23")
    parser.add_argument("--highres_structure_fusion_mode", type=str, default="stage23")
    parser.add_argument("--model_impl", type=str, default="auto", choices=["auto", "standard", "selective"])
    args = parser.parse_args()

    set_deterministic(args.seed)
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

    conn_basis = connectivity_double_angle_basis().to(device)
    score_chunks = {
        "GT ske_q oracle": [],
        "Pred ske_q oracle": [],
        "Pred dir_align oracle": [],
        "Pred ske_q * dir_align": [],
        "Current Con head": [],
    }
    gt_chunks = {name: [] for name in score_chunks}
    per_dir_gt = [[] for _ in range(8)]
    per_dir_scores = {name: [[] for _ in range(8)] for name in score_chunks}
    batches = 0
    images = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Oracle {args.stage}")):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            batches += 1
            images += int(batch["image"].shape[0])

            images_tensor = batch["image"].to(device)
            skeleton_gt = batch["skeleton"].to(device).float()

            outputs = model(images_tensor, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0)
            structure_outputs = outputs[-1]
            selected = select_stage_outputs(structure_outputs)
            if args.stage not in selected:
                raise RuntimeError(f"Stage {args.stage} not present in structure outputs.")
            stage_output = selected[args.stage]

            con_logits = stage_output["connectivity"]
            con_prob = torch.sigmoid(con_logits)
            stage_skeleton_gt = build_stage_skeleton_target(skeleton_gt, con_prob.shape[-2:]).to(device)
            con_gt = build_connectivity_target(stage_skeleton_gt)
            p_skel = stage_skeleton_gt > 0.5

            skeleton_logits, skeleton_source = find_skeleton_logits(
                structure_outputs,
                stage_output,
                outputs[2],
            )
            if batch_idx == 0:
                print(
                    f"[INFO] Pred ske_q oracle skeleton source: {skeleton_source}; "
                    f"selected stage keys={sorted(stage_output.keys())}",
                    flush=True,
                )
            pred_skel = torch.sigmoid(skeleton_logits)
            pred_skel = resize_like(pred_skel, con_prob[:, :1], mode="bilinear").clamp(0.0, 1.0)
            gt_skel = stage_skeleton_gt

            direction_logits = stage_output["direction"]
            direction = F.normalize(direction_logits.float(), dim=1, eps=1e-6)
            dir_align = ((direction.unsqueeze(1) * conn_basis.view(1, 8, 2, 1, 1)).sum(dim=2) + 1.0) * 0.5
            dir_align = dir_align.clamp(0.0, 1.0)

            gt_ske_q = []
            pred_ske_q = []
            for dy, dx in CONNECTIVITY_DIRECTIONS:
                gt_ske_q.append(PairwiseConnectivityHead._shift_feature(gt_skel, dy, dx))
                pred_ske_q.append(PairwiseConnectivityHead._shift_feature(pred_skel, dy, dx))
            gt_ske_q = torch.cat(gt_ske_q, dim=1).clamp(0.0, 1.0)
            pred_ske_q = torch.cat(pred_ske_q, dim=1).clamp(0.0, 1.0)

            score_maps = {
                "GT ske_q oracle": gt_ske_q,
                "Pred ske_q oracle": pred_ske_q,
                "Pred dir_align oracle": dir_align,
                "Pred ske_q * dir_align": pred_ske_q * dir_align,
                "Current Con head": con_prob,
            }

            valid = p_skel.expand_as(con_gt)
            gt_flat = (con_gt[valid] > 0.5).detach().cpu().numpy().astype(np.int32)
            for name, score_map in score_maps.items():
                score_flat = score_map[valid].detach().cpu().numpy().astype(np.float32)
                gt_chunks[name].append(gt_flat)
                score_chunks[name].append(score_flat)

            valid_cpu = p_skel.detach().cpu().bool()
            con_gt_cpu = con_gt.detach().cpu() > 0.5
            for dir_idx in range(8):
                dir_valid = valid_cpu[:, 0]
                dir_gt = con_gt_cpu[:, dir_idx][dir_valid].reshape(-1).numpy().astype(np.int32)
                per_dir_gt[dir_idx].append(dir_gt)
                for name, score_map in score_maps.items():
                    score_cpu = score_map.detach().cpu()
                    per_dir_scores[name][dir_idx].append(
                        score_cpu[:, dir_idx][dir_valid].reshape(-1).numpy().astype(np.float32)
                    )

    print(f"\nCheckpoint: {args.model_path}")
    print(f"split={args.split}, stage={args.stage}, batches={batches}, images={images}")
    print("valid mask: p_skel only")
    print("\nScore source              AUROC    AUPRC    C_pos    C_neg  pos_count  neg_count")
    rows = []
    for name in score_chunks:
        row = summarize(name, gt_chunks[name], score_chunks[name])
        rows.append(row)
        print_row(row)

    print("\nPer-direction AUROC/AUPRC")
    print("dir  positive_ratio " + " ".join(f"{name[:10]:>19s}" for name in score_chunks))
    for dir_idx, dir_name in enumerate(CONNECTIVITY_DIR_NAMES):
        dir_gt = np.concatenate(per_dir_gt[dir_idx], axis=0) if per_dir_gt[dir_idx] else np.empty((0,), dtype=np.int32)
        pos_ratio = float(dir_gt.mean()) if dir_gt.size else float("nan")
        values = []
        for name in score_chunks:
            dir_score = (
                np.concatenate(per_dir_scores[name][dir_idx], axis=0)
                if per_dir_scores[name][dir_idx]
                else np.empty((0,), dtype=np.float32)
            )
            auroc, auprc = safe_auc(dir_gt, dir_score)
            values.append(f"{auroc:8.4f}/{auprc:8.4f}")
        print(f"{dir_name:2s}  {pos_ratio:14.6f} " + " ".join(f"{value:>19s}" for value in values))


if __name__ == "__main__":
    main()
