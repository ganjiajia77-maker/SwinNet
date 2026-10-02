import torch

from .skeleton_guided_head import (
    LegacyConvConnectivityHead,
    LegacyPairwisePriorConnectivityHead,
)
from .skeleton_guided_head_selective_fusion import (
    LegacyConvConnectivityHead as LegacyConvConnectivityHeadSelective,
)


def adapt_connectivity_modules_for_checkpoint(model, state_dict, model_impl):
    keys = [str(key) for key in state_dict.keys()]
    has_pairwise_head = any(".connectivity_head.edge_mlp." in key for key in keys)
    has_legacy_conv_head = any(key.endswith(".connectivity_head.weight") for key in keys)
    has_connectivity_context = any(".connectivity_context." in key for key in keys)

    if not has_connectivity_context:
        for module in model.modules():
            if hasattr(module, "connectivity_context"):
                module.connectivity_context = torch.nn.Identity()

    pairwise_prior_replaced = 0
    if has_pairwise_head and model_impl != "selective":
        divisor = 2
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
                channels = (int(first.in_channels) - prior_channels) // divisor
            legacy_in_channels = divisor * channels + 3
            if int(checkpoint_weight.shape[1]) != legacy_in_channels:
                continue
            connectivity_channels = getattr(head, "connectivity_channels", 8)
            hidden_channels = int(checkpoint_weight.shape[0])
            module.connectivity_head = LegacyPairwisePriorConnectivityHead(
                channels,
                connectivity_channels,
                hidden_channels=hidden_channels,
            ).to(device=first.weight.device, dtype=first.weight.dtype)
            pairwise_prior_replaced += 1
        if pairwise_prior_replaced:
            print(
                "[INFO] Checkpoint uses legacy pairwise prior connectivity heads "
                f"(2C+3); replaced {pairwise_prior_replaced} heads for compatible evaluation.",
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
        connectivity_channels = getattr(head, "connectivity_channels", 8)
        module.connectivity_head = legacy_cls(channels, connectivity_channels).to(
            device=first.weight.device,
            dtype=first.weight.dtype,
        )
        replaced += 1
    if replaced:
        print(
            "[INFO] Checkpoint uses legacy conv connectivity heads; "
            f"replaced {replaced} pairwise heads for compatible evaluation.",
            flush=True,
        )
