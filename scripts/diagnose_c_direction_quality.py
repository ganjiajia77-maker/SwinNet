"""Diagnose 8-direction connectivity C for best_20260728_170803 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import build_connectivity_target
from networks.skeleton_guided_head import VEDRoadDiffusion
from networks.vision_transformer import (
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet as ViT_seg,
)

# Canonical order used by this codebase (NOT clock-wise).
CODE_DIRECTIONS = (
    (-1, 0),  # 0 up
    (1, 0),   # 1 down
    (0, -1),  # 2 left
    (0, 1),   # 3 right
    (-1, -1), # 4 up-left
    (-1, 1),  # 5 up-right
    (1, -1),  # 6 down-left
    (1, 1),   # 7 down-right
)
CODE_OPPOSITE = (1, 0, 3, 2, 7, 6, 5, 4)
CODE_NAMES = (
    "up",
    "down",
    "left",
    "right",
    "up-left",
    "up-right",
    "down-left",
    "down-right",
)

# User-requested clock-wise naming (for mismatch report only).
USER_EXPECTED = (
    "up",
    "up-right",
    "right",
    "down-right",
    "down",
    "down-left",
    "left",
    "up-left",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_path",
        default=r"D:\Code\Swin-Unet-main\model_out\train_0718_final_skeleton_ved_resume\best.pth",
    )
    p.add_argument("--root_path", default=r"D:\Code\Swin-Unet-main\data1")
    p.add_argument("--split", default="test")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--source_patch_size", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_images", type=int, default=0, help="0 = all")
    p.add_argument(
        "--cfg",
        default=r"./configs/swin_tiny_patch4_window7_224_lite.yaml",
    )
    p.add_argument(
        "--output",
        default=r"D:\Code\Swin-Unet-main\predictions\best_20260728_170803\c_direction_diagnostics.txt",
    )
    p.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    # unused config hooks
    p.add_argument("--zip", action="store_true")
    p.add_argument("--cache_mode", type=str, default="")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--accumulation_steps", type=int, default=0)
    p.add_argument("--use_checkpoint", action="store_true")
    p.add_argument("--amp_opt_level", type=str, default="")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--throughput", action="store_true")
    return p.parse_args()


def shift_pad(x, dy, dx):
    """Same semantics as DecoderStructureRefinement._shift_feature."""
    _, _, height, width = x.shape
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    y0 = max(-dy, 0)
    x0 = max(-dx, 0)
    return padded[:, :, y0 : y0 + height, x0 : x0 + width]


def neighbor_roll(x, dy, dx):
    """Same semantics as VEDRoadDiffusion._neighbor: value at (y+dy, x+dx)."""
    shifted = torch.roll(x, shifts=(-dy, -dx), dims=(-2, -1))
    if dy > 0:
        shifted[..., -dy:, :] = 0
    elif dy < 0:
        shifted[..., :(-dy), :] = 0
    if dx > 0:
        shifted[..., :, -dx:] = 0
    elif dx < 0:
        shifted[..., :, :(-dx)] = 0
    return shifted


def mapping_report():
    lines = []
    lines.append("=== 1) 8-direction channel mapping ===")
    lines.append("Code order used by loss / VED / decoder structure:")
    for i, ((dy, dx), name) in enumerate(zip(CODE_DIRECTIONS, CODE_NAMES)):
        lines.append(
            f"  ch{i}: {name:10s}  (dy,dx)=({dy:+d},{dx:+d})  opposite=ch{CODE_OPPOSITE[i]}"
        )
    lines.append("")
    lines.append("User-expected clock-wise order:")
    for i, name in enumerate(USER_EXPECTED):
        lines.append(f"  ch{i}: {name}")
    match = all(CODE_NAMES[i] == USER_EXPECTED[i] for i in range(8))
    lines.append("")
    lines.append(
        f"Match user clock-wise naming? {'YES' if match else 'NO — different index convention'}"
    )
    lines.append(
        "Note: GT connectivity is built with CODE order, so metrics below use CODE order."
    )
    lines.append("")
    lines.append("Diagonal normalization in VEDRoadDiffusion.direction_vectors:")
    for i, (dy, dx) in enumerate(CODE_DIRECTIONS):
        n = math.sqrt(float(dy * dy + dx * dx))
        lines.append(
            f"  ch{i}: unit=({dy/n:.4f},{dx/n:.4f})  norm={n:.4f}  "
            f"{'uses 1/sqrt(2)' if n > 1.1 else 'axis-aligned'}"
        )
    return lines


def synthetic_shift_tests():
    lines = []
    lines.append("=== 2) Synthetic shift / propagation direction tests ===")
    # Impulse at center; see where each shift reads from.
    h = w = 7
    cy = cx = 3
    x = torch.zeros(1, 1, h, w)
    x[0, 0, cy, cx] = 1.0

    lines.append("shift_pad(x, dy, dx)[center] equals value at (center-dy, center-dx):")
    ok_all = True
    for i, (dy, dx) in enumerate(CODE_DIRECTIONS):
        out = shift_pad(x, dy, dx)
        got = float(out[0, 0, cy, cx])
        # Expect 1 only if reading from center means source was at cy+? 
        # out[y,x] = in[y-dy,x-dx]; center gets in[cy-dy,cx-dx]
        # So center=1 iff impulse is at (cy-dy,cx-dx), i.e. this shift pulls FROM opposite of (dy,dx)
        expect_from = "opposite neighbor" if got == 0.0 else "see below"
        # Place impulse at neighbor (cy+dy, cx+dx); check if -dy,-dx shift retrieves it
        x2 = torch.zeros(1, 1, h, w)
        ny, nx = cy + dy, cx + dx
        if 0 <= ny < h and 0 <= nx < w:
            x2[0, 0, ny, nx] = 1.0
        pull_wrong = float(shift_pad(x2, dy, dx)[0, 0, cy, cx])
        pull_right = float(shift_pad(x2, -dy, -dx)[0, 0, cy, cx])
        roll = float(neighbor_roll(x2, dy, dx)[0, 0, cy, cx])
        lines.append(
            f"  ch{i} {CODE_NAMES[i]:10s}: "
            f"shift(+dy,+dx)={pull_wrong:.0f}  "
            f"shift(-dy,-dx)={pull_right:.0f}  "
            f"VED._neighbor={roll:.0f}  "
            f"(want 1 from neighbor at +dir)"
        )
        if pull_right != 1.0 or roll != 1.0:
            ok_all = False
        if pull_wrong == 1.0:
            # decoder directional_propagation uses +dy,+dx — wrong pull
            pass
    lines.append(
        "DecoderStructureRefinement.directional_propagation uses shift(+dy,+dx) "
        "=> pulls OPPOSITE neighbor (high risk if gamma2 != 0)."
    )
    lines.append(
        "VEDRoadDiffusion / SoftGraph-style shift(-dy,-dx) or _neighbor => correct "
        "pull from (dy,dx) neighbor."
    )

    # Artificial C channel = 1 for one direction; observe where mass moves under VED gather.
    lines.append("")
    lines.append("Artificial single-channel C=1 propagation via VED._neighbor gather:")
    feat = torch.zeros(1, 1, h, w)
    feat[0, 0, cy, cx] = 10.0
    for i, (dy, dx) in enumerate(CODE_DIRECTIONS):
        # Simulate: delta += w_d * (neighbor - self); look at where neighbor reads
        neigh = neighbor_roll(feat, dy, dx)
        # Position that receives center's value when gathering neighbor of opposite?
        # neighbor_roll(feat, dy, dx)[y,x] = feat[y+dy,x+dx]
        # So center's 10 appears at pixel (cy-dy, cx-dx): the pixel for which neighbor is center
        target_y, target_x = cy - dy, cx - dx
        val_at_receiver = float(neigh[0, 0, target_y, target_x]) if (
            0 <= target_y < h and 0 <= target_x < w
        ) else float("nan")
        lines.append(
            f"  C_ch{i}={CODE_NAMES[i]:10s}: center mass readable at "
            f"({target_y},{target_x}) from neighbor lookup = {val_at_receiver:.0f}"
        )

    # Line geometry: which channels should fire for GT C
    lines.append("")
    lines.append("GT connectivity on synthetic lines (build_connectivity_target):")
    for name, make in [
        ("horizontal", lambda: _line_mask(h, w, "h")),
        ("vertical", lambda: _line_mask(h, w, "v")),
        ("diag_45_downright", lambda: _line_mask(h, w, "d45")),
        ("diag_135_downleft", lambda: _line_mask(h, w, "d135")),
    ]:
        mask = make()
        c = build_connectivity_target(mask)
        # average C on road pixels
        road = mask > 0.5
        means = []
        for d in range(8):
            m = float(c[:, d : d + 1][road].mean()) if road.any() else 0.0
            means.append(m)
        top = np.argsort(means)[::-1][:3]
        top_str = ", ".join(f"{CODE_NAMES[i]}={means[i]:.2f}" for i in top)
        lines.append(f"  {name}: top channels -> {top_str}")
        if name == "horizontal":
            lr = means[2] + means[3]
            ud = means[0] + means[1]
            lines.append(
                f"    left+right={lr:.3f}, up+down={ud:.3f}  "
                f"({'OK' if lr > ud else 'BAD'})"
            )
        if name == "vertical":
            lr = means[2] + means[3]
            ud = means[0] + means[1]
            lines.append(
                f"    up+down={ud:.3f}, left+right={lr:.3f}  "
                f"({'OK' if ud > lr else 'BAD'})"
            )
    lines.append(f"Synthetic shift sanity overall: {'PASS core neighbor lookups' if ok_all else 'CHECK FAILED'}")
    return lines


def _line_mask(h, w, kind):
    m = torch.zeros(1, 1, h, w)
    c = h // 2
    if kind == "h":
        m[:, :, c, :] = 1
    elif kind == "v":
        m[:, :, :, c] = 1
    elif kind == "d45":
        for i in range(h):
            j = i
            if 0 <= j < w:
                m[:, :, i, j] = 1
    elif kind == "d135":
        for i in range(h):
            j = w - 1 - i
            if 0 <= j < w:
                m[:, :, i, j] = 1
    return m


def load_model(args, device):
    checkpoint = torch.load(args.model_path, map_location="cpu")
    profile = checkpoint.get("structure_profile", STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626)
    saved_args = checkpoint.get("args") if isinstance(checkpoint.get("args"), dict) else {}
    img_size = int(saved_args.get("img_size", args.img_size))
    args.img_size = img_size
    config = get_config(args)
    config.defrost()
    config.DATA.IMG_SIZE = img_size
    config.freeze()

    model = ViT_seg(
        config=config,
        img_size=img_size,
        num_classes=1,
        use_asterisk=True,
        return_skeleton=True,
        bottleneck_type="global_local",
        structure_profile=profile,
        enable_final_graph_prop=False,
    ).to(device)
    state = checkpoint["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    return model, profile, missing, unexpected


@torch.no_grad()
def extract_stage_c(model, images):
    """Return fused stage2/3 skeleton/connectivity logits at full res."""
    outputs = model(images)
    # outputs: surface, boundary, skeleton, connectivity?, + stage stuff depending on return
    # SwinUnet forward returns (logits, *aux). With return_skeleton, aux from guided_head
    # and structure outputs attached via swin_unet last?
    # Better: call swin_unet directly and fuse like up_x4.
    swin = model.swin_unet
    x, x_downsample, road_attentions = swin.forward_features(images)
    x, structure_outputs = swin.forward_up_features(
        x,
        x_downsample,
        bottleneck_tokens=x,
    )
    # full-res size after expand
    H = swin.patches_resolution[0] * 4
    W = swin.patches_resolution[1] * 4
    skel, conn = swin._fuse_stage_structure_to_fullres(
        structure_outputs,
        (H, W),
        stages=(2, 3),
    )
    # also get stage3-only C for reference
    stage_cs = {}
    for item in structure_outputs:
        stage_cs[int(item["stage"])] = F.interpolate(
            item["connectivity"], size=(H, W), mode="bilinear", align_corners=False
        )
        stage_cs[f"skel{int(item['stage'])}"] = F.interpolate(
            item["skeleton"], size=(H, W), mode="bilinear", align_corners=False
        )
    return skel, conn, stage_cs, structure_outputs


def update_prf(stats, pred, gt, threshold=0.5):
    pred_b = pred >= threshold
    gt_b = gt >= 0.5
    stats["tp"] += int((pred_b & gt_b).sum().item())
    stats["fp"] += int((pred_b & ~gt_b).sum().item())
    stats["fn"] += int((~pred_b & gt_b).sum().item())


def prf(stats, eps=1e-7):
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    return p, r, f1


@torch.no_grad()
def evaluate(model, loader, device, max_images=0):
    per_dir = [dict(tp=0, fp=0, fn=0) for _ in range(8)]
    top1_correct = 0
    top1_total = 0
    top2_hit = 0
    top2_total = 0
    esym_sum = 0.0
    esym_count = 0
    # q stats
    q_road_sum = 0.0
    q_road_n = 0
    q_bg_sum = 0.0
    q_bg_n = 0
    active = {
        "road_q01": 0,
        "bg_q01": 0,
        "road_q03": 0,
        "bg_q03": 0,
        "road_n": 0,
        "bg_n": 0,
        "road_c01": 0,
        "bg_c01": 0,
        "road_s01": 0,
        "bg_s01": 0,
    }
    # decoder gamma2
    gammas = []

    n_seen = 0
    ved = model.swin_unet.guided_head.road_diffusion

    for batch in tqdm(loader, desc="eval C"):
        images = batch["image"].to(device)
        mask = batch["mask"].to(device)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        skeleton_gt = batch["skeleton"].to(device)
        if skeleton_gt.dim() == 3:
            skeleton_gt = skeleton_gt.unsqueeze(1)

        skel_logits, conn_logits, stage_cs, _ = extract_stage_c(model, images)
        c_prob = torch.sigmoid(conn_logits)
        s_prob = torch.sigmoid(skel_logits)
        gt_c = build_connectivity_target(skeleton_gt).to(device)

        # resize if needed
        if c_prob.shape[-2:] != mask.shape[-2:]:
            c_prob = F.interpolate(c_prob, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            s_prob = F.interpolate(s_prob, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            gt_c = F.interpolate(gt_c, size=mask.shape[-2:], mode="nearest")

        road = mask > 0.5
        bg = ~road

        for d in range(8):
            update_prf(per_dir[d], c_prob[:, d : d + 1], gt_c[:, d : d + 1])

        # Top-1 / Top-2 on road pixels with any GT connection
        gt_any = gt_c.sum(dim=1, keepdim=True) > 0.5
        valid = road & gt_any
        if valid.any():
            pred_top2 = c_prob.topk(k=2, dim=1)
            pred_top1 = pred_top2.indices[:, 0:1]
            # GT positive set
            gt_pos = gt_c >= 0.5
            # gather whether top1 channel is positive in GT
            top1_ok = torch.gather(gt_pos, 1, pred_top1)
            top1_correct += int(top1_ok[valid].sum().item())
            top1_total += int(valid.sum().item())

            top2_a = torch.gather(gt_pos, 1, pred_top2.indices[:, 0:1])
            top2_b = torch.gather(gt_pos, 1, pred_top2.indices[:, 1:2])
            top2_ok = top2_a | top2_b
            top2_hit += int(top2_ok[valid].sum().item())
            top2_total += int(valid.sum().item())

        # Opposite symmetry on all pixels (or road?)
        for d, (dy, dx) in enumerate(CODE_DIRECTIONS):
            opp = CODE_OPPOSITE[d]
            c_d = c_prob[:, d : d + 1]
            c_opp_shift = neighbor_roll(c_prob[:, opp : opp + 1], dy, dx)
            # E_sym uses |C_i,d - C_{i+u_d},opp|
            diff = (c_d - c_opp_shift).abs()
            esym_sum += float(diff.sum().item())
            esym_count += int(diff.numel())

        # q from VED formula pieces
        skeleton_support = F.max_pool2d(s_prob, 3, 1, 1)
        c_strength = c_prob.topk(k=2, dim=1).values.mean(dim=1, keepdim=True)
        alignment, coherence = ved._directional_alignment(c_prob)
        q = torch.sigmoid(
            ved.confidence(
                torch.cat([skeleton_support, c_strength, coherence], dim=1)
            )
        )

        q_road_sum += float(q[road].sum().item()) if road.any() else 0.0
        q_road_n += int(road.sum().item())
        q_bg_sum += float(q[bg].sum().item()) if bg.any() else 0.0
        q_bg_n += int(bg.sum().item())

        active["road_n"] += int(road.sum().item())
        active["bg_n"] += int(bg.sum().item())
        active["road_q01"] += int((q[road] > 0.1).sum().item()) if road.any() else 0
        active["bg_q01"] += int((q[bg] > 0.1).sum().item()) if bg.any() else 0
        active["road_q03"] += int((q[road] > 0.3).sum().item()) if road.any() else 0
        active["bg_q03"] += int((q[bg] > 0.3).sum().item()) if bg.any() else 0
        active["road_c01"] += int((c_strength[road] > 0.1).sum().item()) if road.any() else 0
        active["bg_c01"] += int((c_strength[bg] > 0.1).sum().item()) if bg.any() else 0
        active["road_s01"] += int((skeleton_support[road] > 0.1).sum().item()) if road.any() else 0
        active["bg_s01"] += int((skeleton_support[bg] > 0.1).sum().item()) if bg.any() else 0

        n_seen += images.size(0)
        if max_images and n_seen >= max_images:
            break

    # decoder gammas
    for idx, block in enumerate(model.swin_unet.decoder_structure_blocks):
        g1 = float(block.gamma1.detach().cpu()) if hasattr(block, "gamma1") else None
        g2 = float(block.gamma2.detach().cpu()) if hasattr(block, "gamma2") else None
        gammas.append((idx, g1, g2))

    return {
        "per_dir": per_dir,
        "top1": (top1_correct, top1_total),
        "top2": (top2_hit, top2_total),
        "esym": (esym_sum, esym_count),
        "q_road": (q_road_sum, q_road_n),
        "q_bg": (q_bg_sum, q_bg_n),
        "active": active,
        "gammas": gammas,
        "n_images": n_seen,
    }


def format_report(map_lines, synth_lines, metrics, profile, missing, unexpected):
    lines = []
    lines.append("C direction quality diagnostics")
    lines.append(f"checkpoint profile: {profile}")
    lines.append(f"images evaluated: {metrics['n_images']}")
    lines.append(f"load missing={len(missing)} unexpected={len(unexpected)}")
    lines.append("")
    lines.extend(map_lines)
    lines.append("")
    lines.extend(synth_lines)
    lines.append("")
    lines.append("=== 3) Per-direction Precision / Recall / F1 (threshold=0.5 on C) ===")
    for d, st in enumerate(metrics["per_dir"]):
        p, r, f1 = prf(st)
        lines.append(
            f"  ch{d} {CODE_NAMES[d]:10s}: P={p:.4f} R={r:.4f} F1={f1:.4f} "
            f"(tp={st['tp']} fp={st['fp']} fn={st['fn']})"
        )
    lines.append("")
    t1c, t1n = metrics["top1"]
    t2c, t2n = metrics["top2"]
    lines.append("=== 4) Top-1 / Top-2 direction accuracy on road pixels with GT edge ===")
    lines.append(
        f"  Top-1 in GT directions: {t1c}/{t1n} = {t1c / max(t1n,1):.4f}"
    )
    lines.append(
        f"  Top-2 contains GT direction: {t2c}/{t2n} = {t2c / max(t2n,1):.4f}"
    )
    es, en = metrics["esym"]
    lines.append("")
    lines.append("=== 5) Opposite consistency E_sym = mean |C_i,d - C_{i+u_d},opp(d)| ===")
    lines.append(f"  E_sym = {es / max(en,1):.6f}")
    lines.append(
        "  (large E_sym => C is poorly antisymmetric; risky as diffusion tensor)"
    )
    lines.append("")
    lines.append("=== 6) Diffusion support q (VED confidence) road vs background ===")
    qr, qrn = metrics["q_road"]
    qb, qbn = metrics["q_bg"]
    lines.append(f"  mean(q | GT road) = {qr / max(qrn,1):.6f}")
    lines.append(f"  mean(q | GT bg)   = {qb / max(qbn,1):.6f}")
    a = metrics["active"]
    lines.append(
        f"  q>0.1 road ratio = {a['road_q01']/max(a['road_n'],1):.4f}  "
        f"bg ratio = {a['bg_q01']/max(a['bg_n'],1):.4f}"
    )
    lines.append(
        f"  q>0.3 road ratio = {a['road_q03']/max(a['road_n'],1):.4f}  "
        f"bg ratio = {a['bg_q03']/max(a['bg_n'],1):.4f}"
    )
    lines.append(
        f"  C_strength>0.1 road = {a['road_c01']/max(a['road_n'],1):.4f}  "
        f"bg = {a['bg_c01']/max(a['bg_n'],1):.4f}"
    )
    lines.append(
        f"  MaxPool3x3(S)>0.1 road = {a['road_s01']/max(a['road_n'],1):.4f}  "
        f"bg = {a['bg_s01']/max(a['bg_n'],1):.4f}"
    )
    lines.append("")
    lines.append("=== 7) Decoder structure gamma (directional residual) ===")
    for idx, g1, g2 in metrics["gammas"]:
        lines.append(f"  stage{idx}: gamma1={g1:.6f} gamma2={g2:.6f}")
    lines.append(
        "  If gamma2≈0, decoder directional_propagation (+dy,+dx bug) has little effect; "
        "VED still uses correct _neighbor."
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    # RTX 50-series may be unsupported by installed CUDA wheels; prefer CPU then.
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            device = torch.device("cuda")
        except RuntimeError:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    map_lines = mapping_report()
    synth_lines = synthetic_shift_tests()

    model, profile, missing, unexpected = load_model(args, device)
    ds = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    if args.max_images:
        ds.image_files = ds.image_files[: args.max_images]
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    metrics = evaluate(model, loader, device, max_images=args.max_images)
    report = format_report(map_lines, synth_lines, metrics, profile, missing, unexpected)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
