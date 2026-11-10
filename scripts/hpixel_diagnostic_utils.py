"""Inference-only probes; no parameters, buffers, or training forward are edited."""
from contextlib import AbstractContextManager
import hashlib
import types

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from datasets.native_relation_targets import native_targets
from topology_direction_constants import CONNECTIVITY_DIRECTIONS


MODES = ('normal', 'c_prior_one', 'residual_off')


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class PixelProbe(AbstractContextManager):
    """Return source weights while preserving the original scalar computations.

    c_prior_one only changes the new pixel module's prior_mode. In particular it
    does not overwrite C3 or remove C from Stage3/global context. residual_off
    uses the existing bypass of the complete added residual (including bias).
    """
    def __init__(self, model):
        self.core = getattr(model, 'swin_unet', model)
        self.module = self.core.h_pixel_reassembly
        if model.training or self.module.training:
            raise ValueError('The whole model must be in eval mode')
        if self.module.prior_mode != 'predicted':
            raise ValueError('Expected a checkpoint trained with the predicted C prior')
        self.original_forward = self.module.forward
        self.had_forward = 'forward' in self.module.__dict__
        self.old_forward = self.module.__dict__.get('forward')
        self.had_disable = hasattr(self.core, 'disable_h_pixel_reassembly')
        self.old_disable = getattr(self.core, 'disable_h_pixel_reassembly', False)
        self.old_prior = self.module.prior_mode
        self.handles = []
        self.reset()

    def reset(self):
        self.inputs = None
        self.native_x = None
        self.base_fusion = None
        self.records = {}
        self.upstream_checks = 0
        self.normal_residual = None
        self.latent = None

    def __enter__(self):
        probe = self

        def record_input(module, args):
            probe.latent = args[0]

        def forward(module, x, h, g, c, return_weights=False):
            current = (x, h, g, c)
            if probe.mode == 'normal':
                probe.inputs = tuple(t.detach().clone() for t in current)
            else:
                if not all(torch.equal(a, b) for a, b in zip(probe.inputs, current)):
                    raise RuntimeError('Upstream x/H/G/C changed between diagnostic modes')
                probe.upstream_checks += 1
            residual, weights = probe.original_forward(x, h, g, c, return_weights=True)
            if probe.mode == 'normal':
                probe.normal_residual = residual.detach().clone()
            spatial = F.conv2d(probe.latent, module.out.weight, bias=None)
            bias = module.out.bias[None, :, None, None]
            if not torch.allclose(spatial + bias, residual, atol=2e-6, rtol=2e-5):
                raise RuntimeError('Output projection decomposition failed')
            probe.records[probe.mode] = {
                'weights': weights[0].detach().cpu().numpy(),
                'residual_abs': residual[0].abs().mean(0).cpu().numpy(),
                'weighted_component_abs': spatial[0].abs().mean(0).cpu().numpy(),
                'residual_change_abs': (residual-probe.normal_residual)[0].abs().mean(0).cpu().numpy(),
                'weighted_component_spatial_std_mean': float(spatial.float().std((-2, -1), unbiased=False).mean()),
            }
            return (residual, weights) if return_weights else residual

        def fusion_input(module, args):
            value = args[0]
            channels = self.module.out.out_channels
            if self.mode == 'normal':
                self.native_x = value[:, :channels].detach().clone()
            elif not torch.equal(self.native_x, value[:, :channels]):
                raise RuntimeError('Native decoder query changed between modes')
            if self.mode == 'residual_off':
                self.base_fusion = value.detach().clone()

        self.handles.append(self.module.out.register_forward_pre_hook(record_input))
        self.handles.append(self.core.surface_h3_fusion.register_forward_pre_hook(fusion_input))
        self.module.forward = types.MethodType(forward, self.module)
        return self

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.module.prior_mode = 'uniform' if mode == 'c_prior_one' else self.old_prior
        self.core.disable_h_pixel_reassembly = mode == 'residual_off'

    def bias_only_tail(self, structure_outputs):
        """Replay only the final head; not a fourth full encoder forward.

        z_normal-z_off = (z_bias_only-z_off)+(z_normal-z_bias_only).
        This is a conditional decomposition, not additive attribution through a
        nonlinear head. A spatially constant feature bias can change logits in
        a spatially varying way.
        """
        if self.base_fusion is None:
            raise RuntimeError('Run residual_off before replaying the tail')
        channels = self.module.out.out_channels
        value = self.base_fusion.clone()
        value[:, channels:] += self.module.out.bias[None, :, None, None]
        h = self.inputs[1]
        result = self.core.guided_head(self.core.surface_h3_fusion(value), z_struct=h)
        result = self.core._apply_structure_surface_correction(result, h, list(structure_outputs))
        return result[0] if isinstance(result, tuple) else result

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        if self.had_forward:
            self.module.forward = self.old_forward
        else:
            del self.module.forward
        self.module.prior_mode = self.old_prior
        if self.had_disable:
            self.core.disable_h_pixel_reassembly = self.old_disable
        else:
            del self.core.disable_h_pixel_reassembly
        self.reset()


def owner_mask(top, left, size=1500, tile=512, stride=256):
    """One disjoint native-pixel partition for overlapping diagnostic windows."""
    from mass_protocol import sliding_positions
    starts = sliding_positions(size, tile, stride)
    centers = np.array(starts, dtype=float) + tile / 2
    bounds = [0] + [int((a+b)/2) for a, b in zip(centers[:-1], centers[1:])] + [size]
    iy, ix = starts.index(top), starts.index(left)
    yy, xx = top + np.arange(tile), left + np.arange(tile)
    return ((yy[:, None] >= bounds[iy]) & (yy[:, None] < bounds[iy+1]) &
            (xx[None, :] >= bounds[ix]) & (xx[None, :] < bounds[ix+1]))


def local_closed(pred, gap, margin=16):
    pixels = np.asarray(gap['pixels'], dtype=np.int64)
    if not pixels.size or pred.ravel()[pixels].mean() < .8:
        return False
    anchors = [np.asarray(a, dtype=np.int64) for a in gap['anchors']]
    if len(anchors) < 2 or any(not len(a) for a in anchors):
        return False
    points = np.concatenate([pixels] + anchors)
    yy, xx = np.divmod(points, pred.shape[1])
    y0, y1 = max(int(yy.min())-margin, 0), min(int(yy.max())+margin+1, pred.shape[0])
    x0, x1 = max(int(xx.min())-margin, 0), min(int(xx.max())+margin+1, pred.shape[1])
    _, labels = cv2.connectedComponents(pred[y0:y1, x0:x1].astype(np.uint8), connectivity=8)
    groups = []
    for anchor in anchors:
        ay, ax = np.divmod(anchor, pred.shape[1])
        groups.append(set(labels[ay-y0, ax-x0]) - {0})
    return bool(set.intersection(*groups))


def difficult_background(road):
    """Background within cells incident to valid GT-negative road/road edges.

    C labels are exactly the original B generator. A zero skeleton skips the
    unused direction tracing; neither connectivity labels nor masks depend on
    the skeleton. These regions are a proxy, not verified false bridges.
    """
    labels = native_targets(road, np.zeros_like(road))
    h, w = labels['ambiguous_cells'].shape
    occupied = road.reshape(h, 4, w, 4).any(axis=(1, 3))
    padded = np.pad(occupied, 1)
    cells = np.zeros_like(occupied)
    for d, (dy, dx) in enumerate(CONNECTIVITY_DIRECTIONS):
        neighbour = padded[1+dy:1+dy+h, 1+dx:1+dx+w]
        edge = (labels['connectivity_valid'][d] & (labels['connectivity'][d] < .5) &
                occupied & neighbour)
        cells |= edge
        yy, xx = np.nonzero(edge)
        cells[yy+dy, xx+dx] = True
    return cells.repeat(4, 0).repeat(4, 1) & ~road


def distribution(values, mask):
    selected = np.asarray(values)[mask].astype(np.float64)
    if not selected.size:
        return dict(pixels=0, signed_mean=None, abs_mean=None, abs_max=None,
                    p10=None, median=None, p90=None, increased=0, decreased=0)
    q = np.quantile(selected, [.1, .5, .9])
    return dict(pixels=int(selected.size), signed_mean=float(selected.mean()),
                abs_mean=float(np.abs(selected).mean()), abs_max=float(np.abs(selected).max()),
                p10=float(q[0]), median=float(q[1]), p90=float(q[2]),
                increased=int((selected > 0).sum()), decreased=int((selected < 0).sum()))
