"""Checkpoint compatibility used by validation and test entry points."""

import torch

from .skeleton_guided_head import (
    PairwiseConnectivityHead,
    LegacyConvConnectivityHead,
    LegacyPairwisePriorConnectivityHead,
)
from .skeleton_guided_head_selective_fusion import (
    LegacyConvConnectivityHead as LegacyConvConnectivityHeadSelective,
)


def adapt_connectivity_modules_for_checkpoint(model, state_dict, model_impl):
    # Reject baseline prior-conditioned heads instead of silently reinitializing C.
    for name, head in model.named_modules():
        if not isinstance(head, PairwiseConnectivityHead):
            continue
        key = f"{name}.edge_linear.weight"
        weight = state_dict.get(key)
        if weight is None or weight.shape != head.edge_linear.weight.shape:
            raise ValueError(f"Expected feature-pair-only connectivity weights: {key}")
        if any(key.startswith(name + '.prior_embed.') for key in state_dict):
            raise ValueError(f"S/D-conditioned checkpoint cannot load into this ablation: {name}")
    keys = [str(key) for key in state_dict]
    has_pairwise_head = any(".connectivity_head.edge_mlp." in key for key in keys)
    has_legacy_conv_head = any(key.endswith(".connectivity_head.weight") for key in keys)
    has_connectivity_context = any(".connectivity_context." in key for key in keys)

    if not has_connectivity_context:
        for module in model.modules():
            if hasattr(module, "connectivity_context"):
                module.connectivity_context = torch.nn.Identity()

    if has_pairwise_head and model_impl != "selective":
        replaced = 0
        for module_name, module in model.named_modules():
            head = getattr(module, "connectivity_head", None)
            if head is None or not hasattr(head, "edge_mlp"):
                continue
            key = f"{module_name}.connectivity_head.edge_mlp.0.weight"
            checkpoint_weight = state_dict.get(key)
            if checkpoint_weight is None:
                continue
            first = head.edge_mlp[0]
            if int(checkpoint_weight.shape[1]) == int(first.in_channels):
                continue
            channels = int(getattr(head, "feature_channels", 0))
            if channels <= 0:
                prior_channels = int(getattr(head, "prior_channels", 16))
                channels = (int(first.in_channels) - prior_channels) // 2
            if int(checkpoint_weight.shape[1]) != 2 * channels + 3:
                continue
            module.connectivity_head = LegacyPairwisePriorConnectivityHead(
                channels,
                getattr(head, "connectivity_channels", 8),
                hidden_channels=int(checkpoint_weight.shape[0]),
            ).to(device=first.weight.device, dtype=first.weight.dtype)
            replaced += 1
        if replaced:
            print(
                f"[INFO] Restored {replaced} legacy pairwise connectivity heads.",
                flush=True,
            )

    if not has_legacy_conv_head or has_pairwise_head:
        return

    legacy_cls = (
        LegacyConvConnectivityHeadSelective
        if model_impl == "selective"
        else LegacyConvConnectivityHead
    )
    divisor = 4 if model_impl == "selective" else 2
    replaced = 0
    for module in model.modules():
        head = getattr(module, "connectivity_head", None)
        if head is None or not hasattr(head, "edge_mlp"):
            continue
        first = head.edge_mlp[0]
        channels = int(getattr(head, "feature_channels", 0))
        if channels <= 0:
            prior_channels = int(getattr(head, "prior_channels", 3))
            channels = (int(first.in_channels) - prior_channels) // divisor
        module.connectivity_head = legacy_cls(
            channels, getattr(head, "connectivity_channels", 8)
        ).to(device=first.weight.device, dtype=first.weight.dtype)
        replaced += 1
    if replaced:
        print(f"[INFO] Restored {replaced} legacy conv connectivity heads.", flush=True)
