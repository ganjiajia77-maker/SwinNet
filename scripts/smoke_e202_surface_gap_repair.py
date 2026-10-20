#!/usr/bin/env python3
"""Run one 512px base-model and surface-repair forward pass before training."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surface_gap_repair_experiment import base_forward, build_base
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


if __name__ == "__main__":
    main()
