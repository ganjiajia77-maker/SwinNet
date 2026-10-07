"""Frozen DINOv2-L/16 features with the original H0 road-aware pyramid."""

import contextlib
import math
import re
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import VisionTransformer


def convert_dino_state(checkpoint, target):
    state = checkpoint
    while isinstance(state, dict):
        key = next((k for k in ("state_dict", "model", "teacher") if isinstance(state.get(k), dict)), None)
        if key is None:
            break
        state = state[key]
    if not isinstance(state, dict):
        raise ValueError("Expected a DINOv2 tensor state dictionary")
    converted = {}
    ignored = []
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        name = key
        while name.startswith(("module.", "backbone.", "teacher.")):
            name = name.split(".", 1)[1]
        if name == "mask_token" or name.startswith(("head.", "dino_head.", "ibot_head.")):
            ignored.append(name)
            continue
        name = name.replace("patch_embed.projection.", "patch_embed.proj.")
        name = re.sub(r"^layers\.(\d+)\.", r"blocks.\1.", name)
        for old, new in ((".ln1.", ".norm1."), (".ln2.", ".norm2."),
                         (".attn.attn.in_proj_weight", ".attn.qkv.weight"),
                         (".attn.attn.in_proj_bias", ".attn.qkv.bias"),
                         (".attn.attn.out_proj.", ".attn.proj."),
                         (".ffn.layers.0.0.", ".mlp.fc1."),
                         (".ffn.layers.1.", ".mlp.fc2.")):
            name = name.replace(old, new)
        if name.startswith("ln1."):
            name = "norm." + name[4:]
        if name in converted:
            raise ValueError(f"Multiple source tensors map to {name}")
        converted[name] = value

    kernel = converted.get("patch_embed.proj.weight")
    expected_kernel = target["patch_embed.proj.weight"]
    if kernel is None or kernel.shape != expected_kernel.shape:
        raise ValueError(
            f"DINOv2-L/16 requires patch kernel {tuple(expected_kernel.shape)}; "
            f"got {None if kernel is None else tuple(kernel.shape)}. "
            "Official patch14 weights are not patch16 converted weights."
        )
    source_pos = converted.get("pos_embed")
    target_pos = target["pos_embed"]
    if source_pos is not None and source_pos.shape != target_pos.shape:
        if source_pos.ndim != 3 or source_pos.shape[:1] != (1,) or source_pos.shape[2] != target_pos.shape[2]:
            raise ValueError("Invalid DINOv2 positional embedding shape")
        source_side = math.isqrt(source_pos.shape[1] - 1)
        target_side = math.isqrt(target_pos.shape[1] - 1)
        if source_side ** 2 != source_pos.shape[1] - 1:
            raise ValueError("Expected one class token followed by a square patch grid")
        spatial = source_pos[:, 1:].reshape(1, source_side, source_side, -1).permute(0, 3, 1, 2)
        spatial = F.interpolate(spatial.float(), (target_side, target_side), mode="bicubic", align_corners=False)
        converted["pos_embed"] = torch.cat((source_pos[:, :1].float(), spatial.flatten(2).transpose(1, 2)), dim=1)

    missing = sorted(set(target) - set(converted))
    unexpected = sorted(set(converted) - set(target))
    mismatched = [name for name in set(target) & set(converted) if target[name].shape != converted[name].shape]
    if missing or unexpected or mismatched:
        raise RuntimeError(f"Incomplete/incompatible frozen DINOv2 weights: missing={missing}, unexpected={unexpected}, shape={mismatched}")
    nonfinite = [name for name, value in converted.items() if not torch.isfinite(value).all()]
    if nonfinite:
        raise ValueError(f"Non-finite pretrained tensors: {nonfinite}")
    return converted, ignored


class DinoRoadEncoder(nn.Module):
    def __init__(self, image_size, embed_dim, merges, road_stage, frozen=True, backbone=None):
        super().__init__()
        if image_size % 32:
            raise ValueError("H0 DINO input size must be divisible by 32")
        self.frozen = bool(frozen)
        self.image_size = image_size
        self.backbone = backbone if backbone is not None else VisionTransformer(
            img_size=image_size, patch_size=16, in_chans=3, num_classes=0,
            embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
            qkv_bias=True, proj_bias=True, init_values=1e-5,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not self.frozen)
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1024, embed_dim * 2 ** i, 1), nn.GroupNorm(1, embed_dim * 2 ** i))
            for i in range(4)
        ])
        self.merges = nn.ModuleList(merges)
        self.road_stage = road_stage
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def load_pretrained(self, path):
        # Converted weights supplied by the user must come from a trusted source.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state, ignored = convert_dino_state(checkpoint, self.backbone.state_dict())
        self.backbone.load_state_dict(state, strict=True)
        print(f"[DINO] Loaded all {len(state)} backbone tensors; ignored unused heads/tokens: {ignored}; frozen={self.frozen}", flush=True)
        return {"swin_unet.dino_encoder.backbone." + name for name in state}

    def forward(self, image, road_heads):
        if image.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"DINO/H0 expects {self.image_size} square tiles, got {tuple(image.shape)}")
        context = torch.no_grad() if self.frozen else contextlib.nullcontext()
        with context:
            features = self.backbone.forward_intermediates(
                image, indices=[5, 11, 17, 23], norm=True,
                intermediates_only=True, output_fmt="NCHW",
            )
        pyramid = []
        for i, (feature, projection) in enumerate(zip(features, self.projections)):
            side = self.image_size // (4 * 2 ** i)
            feature = F.interpolate(projection(feature), (side, side), mode="bilinear", align_corners=False)
            pyramid.append(feature.flatten(2).transpose(1, 2))
        x = pyramid[0]
        skips, road_attentions, road_maps = [], [], []
        for i in range(2):
            skips.append(x)
            side = self.image_size // (4 * 2 ** i)
            roadmap = road_heads[i](x.transpose(1, 2).reshape(image.shape[0], -1, side, side))
            road_maps.append(roadmap)
            road_attentions.append({"stage": f"encoder_stage{i + 1}_road_attention", "road_attention": roadmap})
            x = self.merges[i](x, road_attention=roadmap) + pyramid[i + 1]
        skips.append(x)
        x = self.road_stage(x, road_prior=tuple(road_maps)) + pyramid[3]
        skips.append(x)
        return x, skips, road_attentions
