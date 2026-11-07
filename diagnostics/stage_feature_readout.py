"""Read-only stage hooks and full-channel linear road probes for native B."""
from collections import OrderedDict
import math
import torch
from torch import nn
import torch.nn.functional as F


def token_map(value):
    if value.ndim != 3:
        raise ValueError(f'Expected BLC tokens, got {value.shape}')
    side = math.isqrt(value.shape[1])
    if side*side != value.shape[1]:
        raise ValueError('This diagnostic requires square tokens')
    return value.transpose(1, 2).reshape(value.shape[0], value.shape[2], side, side)


class StageCapture:
    """Hooks return None and never replace module inputs/outputs."""
    def __init__(self, model):
        self.core = model.swin_unet
        self.values = OrderedDict()
        self.handles = []
        self.metadata = OrderedDict()
        self.active = True
        core = self.core

        def output(module, name, tokens=False):
            def hook(m, args, result):
                if self.active:self.save(name, token_map(result) if tokens else result)
            self.handles.append(module.register_forward_hook(hook))

        def before(module, name, tokens=False):
            def hook(m, args):
                if self.active:self.save(name, token_map(args[0]) if tokens else args[0])
            self.handles.append(module.register_forward_pre_hook(hook))

        output(core.patch_embed, 'patch_embed', True)
        for i, layer in enumerate(core.layers, 1):
            if layer.downsample is not None:
                before(layer.downsample, f'enc{i}_blocks_before_down', True)
                output(layer.downsample, f'enc{i}_after_down', True)
            else:
                output(layer, f'enc{i}_after_blocks', True)
        output(core.bottleneck_context_fusion, 'bottleneck_context', True)
        output(core.bottleneck_swin_block, 'bottleneck_transformer', True)
        output(core.norm, 'bottleneck_norm', True)
        for i, block in enumerate(core.dca_blocks, 2):
            output(block, f'dec{i}_DCA_refined_skip')
        for i, layer in enumerate(core.layers_up):
            before(layer, f'dec{i}_input_after_skip' if i else 'dec0_input_no_skip', True)
            if hasattr(layer, 'upsample') and layer.upsample is not None:
                before(layer.upsample, f'dec{i}_blocks_before_up', True)
                output(layer.upsample, f'dec{i}_after_up', True)
            else:
                output(layer, f'dec{i}_after_blocks' if hasattr(layer, 'blocks') else f'dec{i}_after_up', True)
        for i in (2, 3):
            def hook(m, args, result, index=i):
                if not self.active:return
                self.save(f'dec{index}_after_structure', result[0])
                info = result[-1]
                self.save(f'G{index}', info['structure_feat'])
                if index == 3:
                    self.save('H_stage3_before_relation', info['h2_before_relation'])
                    self.save('H_final_after_relation', info['highres_skeleton_feat'])
            self.handles.append(core.decoder_structure_blocks[str(i)].register_forward_hook(hook))
        before(core.native_global_context, 'H_stage2_before_global')
        output(core.native_global_context, 'H_stage2_after_global')
        output(core.highres_skeleton_adapter, 'H0')
        output(core.norm_up, 'decoder_final_norm', True)
        output(core.up, 'final_x4_after_up', True)
        output(core.surface_h3_fusion, 'final_H_G3_surface_fusion')
        output(core.guided_head.surface_refine, 'surface_refine_before_boundary')
        before(core.guided_head.surface_head, 'surface_head_input_after_boundary')

    def save(self, name, value):
        if value.ndim != 4:
            raise ValueError(f'{name}: expected BCHW, got {value.shape}')
        self.values[name] = value.detach().clone()
        shape = list(value.shape[1:])
        if name in self.metadata and self.metadata[name] != shape:
            raise RuntimeError(f'Stage shape changed: {name}')
        self.metadata[name] = shape

    def clear(self):
        self.values.clear()

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles.clear()
        self.clear()


class LinearReadout(nn.Module):
    """No channel truncation; train-only channel standardization, then 1x1."""
    def __init__(self, channels):
        super().__init__()
        self.register_buffer('mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('std', torch.ones(1, channels, 1, 1))
        self.head = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feature, size=None):
        z = self.head((feature-self.mean)/self.std)
        return F.interpolate(z, size=size, mode='bilinear', align_corners=False) if size else z


def road_loss(z, target):
    bce = F.binary_cross_entropy_with_logits(z, target)
    prob = z.sigmoid()
    dice = 1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1)
    return bce+dice


def stage_name_order(names):
    # Actual execution order, recorded by hooks. Not lexicographic ordering.
    return list(names)
