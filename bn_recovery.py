"""Separate MSWD running statistics while preserving its learned affine parameters."""

import copy

from networks.WeavingUnet import MSWD


METHOD = "mswd_split_stats_train_only_layerwise_v1"


def split_mswd_stats(model):
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, MSWD):
            continue
        if module.norm_second is None:
            module.norm_second = copy.deepcopy(module.norm)
            module.norm_second.weight = module.norm.weight
            module.norm_second.bias = module.norm.bias
        names.extend((f"{name}.norm", f"{name}.norm_second"))
    return names


def load_checkpoint_model(model, checkpoint):
    recovery = checkpoint.get("bn_recovery")
    if recovery is not None:
        if recovery.get("method") != METHOD:
            raise ValueError(f"Unknown BN recovery method: {recovery.get('method')}")
        split_mswd_stats(model)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model
