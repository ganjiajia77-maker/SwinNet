"""Smoke test for direction-aware graph diffusion integration."""

from __future__ import annotations

import argparse

import torch

from config import get_config
from networks.graph_diffusion import DirectionAwareGraphDiffusion
from networks.vision_transformer import (
    TOPOLOGY_ATTENTION_VERSION,
    SwinUnet,
    load_topology_checkpoint_state,
)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="./data1")
    parser.add_argument("--dataset", default="ImageData")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--output_dir", default="./model_out")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument(
        "--cfg",
        default="./configs/swin_tiny_patch4_window7_224_lite.yaml",
    )
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--zip", action="store_true", default=False)
    parser.add_argument("--cache_mode", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--accumulation_steps", type=int, default=0)
    parser.add_argument("--use_checkpoint", action="store_true", default=False)
    parser.add_argument("--amp_opt_level", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--throughput", action="store_true", default=False)
    return parser.parse_args([])


def test_graph_diffusion_module():
    module = DirectionAwareGraphDiffusion(channels=32)
    feature = torch.randn(2, 32, 16, 16)
    skeleton = torch.sigmoid(torch.randn(2, 1, 16, 16))
    connectivity = torch.sigmoid(torch.randn(2, 8, 16, 16))
    direction = torch.randn(2, 2, 16, 16)
    out, message = module.diffuse(feature, skeleton, connectivity, direction)
    assert out.shape == feature.shape
    assert message.shape == feature.shape
    assert torch.isfinite(out).all()
    assert out.requires_grad
    loss = out.sum()
    loss.backward()
    assert module.gamma.grad is not None


def test_model_forward():
    args = build_args()
    config = get_config(args)
    model = SwinUnet(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        return_skeleton=True,
        structure_profile="full",
        enable_graph_diffusion=True,
        enable_structure_gate=False,
        enable_decoder_attention_bias=False,
    )
    x = torch.randn(1, 3, args.img_size, args.img_size)
    outputs = model(x)
    assert isinstance(outputs, tuple)
    assert outputs[0].shape == (1, 1, args.img_size, args.img_size)
    assert TOPOLOGY_ATTENTION_VERSION == "direction-graph-diffusion-v1"


def test_checkpoint_compat():
    args = build_args()
    config = get_config(args)
    baseline = SwinUnet(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        return_skeleton=True,
        structure_profile="full",
        enable_graph_diffusion=False,
        enable_structure_gate=True,
        enable_decoder_attention_bias=True,
    )
    state = baseline.state_dict()
    graph_model = SwinUnet(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        return_skeleton=True,
        structure_profile="full",
        enable_graph_diffusion=True,
        enable_structure_gate=False,
        enable_decoder_attention_bias=False,
    )
    result = load_topology_checkpoint_state(
        graph_model,
        state,
        "legacy-test",
        strict=True,
    )
    missing_graph = [
        key for key in result.missing_keys if "graph_diffusion" in key
    ]
    assert missing_graph, "graph diffusion params should be newly initialized"
    invalid_missing = [
        key
        for key in result.missing_keys
        if "graph_diffusion" not in key
    ]
    assert not invalid_missing, invalid_missing


def main():
    test_graph_diffusion_module()
    test_model_forward()
    test_checkpoint_compat()
    print("[PASS] graph diffusion verification succeeded.")


if __name__ == "__main__":
    main()
