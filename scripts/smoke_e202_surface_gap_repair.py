#!/usr/bin/env python3
"""Run one 512px base-model and surface-repair forward pass before training."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surface_gap_repair_experiment import (
    base_forward, build_base, focal_dice, masked_soft_cldice, online_repair_forward,
)
from networks.surface_gap_repair import GapCandidate, SurfaceGapRepair, candidate_tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cfg", default=str(ROOT / "configs/swin_tiny_patch4_window7_224_lite.yaml"))
    parser.add_argument(
        "--pretrain_ckpt",
        default="/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this preflight with the training environment and GPU")

    device = torch.device(f"cuda:{args.gpu}")
    model_args = argparse.Namespace(cfg=args.cfg, pretrain_ckpt=args.pretrain_ckpt)
    model, _ = build_base(model_args, device, load_imagenet=False)
    model.eval()
    image = torch.zeros(1, 3, 512, 512, device=device)
    with torch.inference_mode():
        logits, features = base_forward(model, image, features=True)

    expected = {
        "logits": (1, 1, 512, 512),
        "encoder_stage1": (1, 96, 128, 128),
        "decoder_final": (1, 96, 512, 512),
    }
    actual = {"logits": tuple(logits.shape)}
    actual.update({key: tuple(value.shape) for key, value in features.items()})
    if actual != expected:
        raise RuntimeError(f"Base model feature contract mismatch: got {actual}, expected {expected}")

    candidate = GapCandidate(
        kind="endpoint_endpoint",
        anchor_a=(200, 200), anchor_b=(220, 250),
        tangent_a=(0.0, 1.0), tangent_a_valid=True,
        tangent_b=(0.0, -1.0), tangent_b_valid=True,
        distance=53.85, line_alignment_a=0.9, line_alignment_b=0.9,
        tile_top=0, tile_left=0,
    )
    prompts, geometry = candidate_tensors(candidate, device)
    if tuple(prompts.shape) != (1, 2, 512, 512) or tuple(geometry.shape) != (1, 8):
        raise RuntimeError(
            f"Candidate tensor contract mismatch: prompts={tuple(prompts.shape)}, "
            f"geometry={tuple(geometry.shape)}"
        )

    repair = SurfaceGapRepair().to(device).eval()
    with torch.inference_mode():
        cls, delta = repair(
            image, torch.sigmoid(logits), prompts, geometry,
            features["encoder_stage1"], features["decoder_final"],
        )
    if tuple(cls.shape) != (1,) or tuple(delta.shape) != (1, 1, 512, 512):
        raise RuntimeError(f"Repair output contract mismatch: cls={tuple(cls.shape)}, delta={tuple(delta.shape)}")
    if float(delta.abs().max()) != 0.0:
        raise RuntimeError("Zero-initialized repair delta is not zero")
    print(
        "E202_SMOKE_OK "
        f"logits={tuple(logits.shape)} stage1={tuple(features['encoder_stage1'].shape)} "
        f"decoder={tuple(features['decoder_final'].shape)} geometry={tuple(geometry.shape)} "
        f"cls={tuple(cls.shape)} delta={tuple(delta.shape)}",
        flush=True,
    )

    del model, image, logits, features, repair
    torch.cuda.empty_cache()

    # Exercise the same candidate generation, labeling, repair and loss path as one
    # training batch, using a synthetic broken road so a candidate is guaranteed.
    train_image = torch.zeros(1, 3, 512, 512, device=device)
    target = torch.zeros(1, 1, 512, 512, device=device)
    target[0, 0, 256, 40:440] = 1.0
    base_logits = torch.full((1, 1, 512, 512), -8.0, device=device)
    base_logits[0, 0, 250:263, 80:210] = 4.0
    base_logits[0, 0, 250:263, 226:390] = 4.0
    base_logits.requires_grad_()
    train_features = {
        "encoder_stage1": torch.randn(1, 96, 128, 128, device=device, requires_grad=True),
        "decoder_final": torch.randn(1, 96, 512, 512, device=device, requires_grad=True),
    }
    train_repair = SurfaceGapRepair().to(device).train()
    final_logits, connection_loss, counts, records, seen = online_repair_forward(
        train_repair, train_image, target, base_logits, train_features, 0.45,
    )
    if seen < 1 or not any(row["label"] == 1 for row in records):
        raise RuntimeError(f"Synthetic broken-road candidate was not positively labeled: {records}")
    safe = torch.ones_like(target)
    safe[:, :, :32, :] = 0
    safe[:, :, -32:, :] = 0
    safe[:, :, :, :32] = 0
    safe[:, :, :, -32:] = 0
    cldice_loss = masked_soft_cldice(final_logits, target, safe)
    loss = focal_dice(final_logits, target) + 0.1 * cldice_loss + connection_loss
    loss.backward()
    if not torch.isfinite(loss) or train_repair.classifier[-1].weight.grad is None:
        raise RuntimeError("Training-path loss or classifier gradient is invalid")
    if train_repair.delta512[-1].weight.grad is None or base_logits.grad is None:
        raise RuntimeError("Training-path surface gradient is missing")
    print(
        "E202_TRAIN_PATH_OK "
        f"candidates={seen} valid_labels={sum(row['label'] >= 0 for row in records)} "
        f"surface={float(focal_dice(final_logits.detach(), target)):.6f} "
        f"cldice={float(cldice_loss.detach()):.6f} "
        f"connection={float(connection_loss.detach()):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
