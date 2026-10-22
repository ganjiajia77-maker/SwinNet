"""Finite synthetic verification only. Does not start a dataset training run."""
import argparse
import gc
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from networks.encoder_direction import E2DirectionAggregation
from losses.encoder_direction_loss import direction_target, encoder_direction_loss, DEFAULT_DIRECTION_CONFIG
from losses.road_losses import SurfaceStructureLoss
from h0_encoder_protocol import stitch_logits
from scripts.verify_h0_encoder import make_model, verify_persistent_worker_epoch_sync


def module_checks(device):
    torch.manual_seed(1234)
    module = E2DirectionAggregation(8).to(device)
    feature = torch.randn(2, 8, 9, 11, device=device, requires_grad=True)
    initial, _ = module(feature)
    assert torch.equal(initial, feature)
    assert abs(float(module.gamma) - .05) < 1e-7
    tangent = F.normalize(torch.randn(2, 2, 9, 11, device=device), dim=1)
    weights = torch.randn(2, 2, 9, 11, device=device).softmax(1)
    positive, valid = module.aggregate(feature, tangent, weights)
    negative, valid_neg = module.aggregate(feature, -tangent, weights)
    assert torch.equal(valid, valid_neg)
    assert torch.allclose(positive, negative, atol=1e-6, rtol=1e-6)
    for value in (0., 1e-15, 1e-5):
        raw = torch.full((1, 2, 5, 5), value, device=device, requires_grad=True)
        module.tangent(raw).sum().backward()
        assert torch.isfinite(raw.grad).all()
    # No valid symmetric neighbor on singleton images, even after a learned bias.
    with torch.no_grad():
        module.projection.bias.fill_(1)
    point = torch.randn(2, 8, 1, 1, device=device)
    assert torch.equal(module(point)[0], point)
    with torch.no_grad():
        module.projection.bias.zero_()
    mask = torch.zeros(2, 1, 64, 64, device=device)
    skel = torch.zeros_like(mask)
    mask[:, :, 28:37, 4:60] = 1
    skel[:, :, 32, 4:60] = 1
    target, reliability = direction_target(skel, mask, (8, 8), DEFAULT_DIRECTION_CONFIG)
    selected = reliability[:, 0] > 0
    assert torch.allclose(target[:, 0][selected], torch.ones_like(target[:, 0][selected]), atol=1e-5)
    assert torch.allclose(target[:, 1][selected], torch.zeros_like(target[:, 1][selected]), atol=1e-5)
    vertical, vertical_weight = direction_target(skel.transpose(2, 3), mask.transpose(2, 3), (8, 8), DEFAULT_DIRECTION_CONFIG)
    assert (vertical[:, 0][vertical_weight[:, 0] > 0] < -.99).all()
    raw = torch.zeros(2, 2, 8, 8, device=device, requires_grad=True)
    empty_loss, count, _ = encoder_direction_loss(raw, skel*0, mask*0, DEFAULT_DIRECTION_CONFIG)
    empty_loss.backward()
    assert float(empty_loss) == float(count) == 0 and torch.isfinite(raw.grad).all()
    optimizer = torch.optim.SGD(module.parameters(), lr=.5)
    destination = torch.randn_like(feature)
    records = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        out, _ = module(feature)
        F.mse_loss(out, destination).backward()
        grad = lambda block: sum(float(p.grad.norm()) for p in block.parameters() if p.grad is not None)
        records.append({'step': step, 'D': grad(module.direction_head),
                        'q': grad(module.gate_head), 'projection': grad(module.projection)})
        assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
        optimizer.step()
    assert records[0]['q'] == 0 and records[1]['q'] > 0 and records[1]['D'] > 0
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    print('MODULE_CHECKS_OK ' + json.dumps(records), flush=True)


def full_checks(device, batch_sizes, native):
    torch.manual_seed(1234)
    model = make_model(True).to(device)
    assert not any('direction' in name for name in model.swin_unet.decoder_structure_blocks.state_dict())
    assert model.swin_unet.stage2_topology_source is None
    criterion = SurfaceStructureLoss(
        surface_dice_weight=.5, skeleton_dice_weight=1.,
        stage_structure_weights=(0, 0, .008, .012),
        stage_skeleton_only_loss_factor=.8, stage_skeleton_loss_factor=1.,
        stage_connectivity_factor=3., stage_direction_factor=0.,
        skeleton_pos_weight=2., connectivity_pos_weight=2.,
        use_masked_connectivity_center_experiment=True,
        connectivity_focal_gamma=1.5, surface_focal_gamma=1.,
        encoder_road_aux_weight=.1, encoder_skeleton_aux_weight=.004,
    ).to(device)
    optimizer = torch.optim.SGD(model.swin_unet.e2_direction.parameters(), lr=.1)
    projection_updated = False
    order, hooks = [], []
    for name, block in [('swin2', model.swin_unet.layers[1].blocks[-1]),
                        ('aggregate', model.swin_unet.e2_direction),
                        ('exchange2', model.swin_unet.encoder_exchanges[1]),
                        ('merge2', model.swin_unet.layers[1].downsample)]:
        hooks.append(block.register_forward_hook(lambda m, x, y, name=name: order.append(name)))
    for batch in batch_sizes:
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model.train()
        model.zero_grad(set_to_none=True)
        image = torch.randn(batch, 3, 512, 512, device=device)
        target = torch.zeros(batch, 1, 512, 512, device=device)
        skeleton = torch.zeros_like(target)
        target[:, :, 240:270, 80:430] = 1
        skeleton[:, :, 255, 80:430] = 1
        order.clear()
        output = model(image)
        assert order == ['swin2', 'aggregate', 'exchange2', 'merge2'], order
        assert output[0].shape == (batch, 1, 512, 512)
        assert model.swin_unet.last_encoder_features['E2'].shape == (batch, 192, 64, 64)
        stages = [item for item in output[4] if item.get('stage') in (2, 3)]
        assert len(stages) == 2
        assert all(item.get('direction') is None and item['connectivity'].shape[1] == 8 for item in stages)
        loss, parts = criterion(output[0], surface_gt=target, skeleton_gt=skeleton,
            skeleton_dilate_gt=F.max_pool2d(skeleton, 3, 1, 1),
            stage_outputs=output[4], encoder_aux_outputs=output[5],
            boundary_logits=output[1], skeleton_logits=output[2], connectivity_logits=output[3])
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        gate_grad = sum(float(p.grad.abs().sum()) for p in model.swin_unet.e2_direction.gate_head.parameters()
                        if p.grad is not None)
        if projection_updated:
            assert gate_grad > 0, 'q must receive original-task gradients after projection update'
        for prefix in ('e2_direction.direction_head', 'e2_direction.projection',
                       'encoder_road_heads', 'encoder_skeleton_heads', 'encoder_exchanges',
                       'decoder_structure_blocks.2.skeleton_head', 'decoder_structure_blocks.3.connectivity_head'):
            params = [p for n, p in model.swin_unet.named_parameters() if n.startswith(prefix)]
            assert params and any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in params), prefix
        print('FULL_FORWARD_BACKWARD_OK ' + json.dumps({'batch': batch,
            'loss': float(loss), 'direction': float(parts['encoder_direction_raw']),
            'q_grad_l1': gate_grad,
            'valid': int(parts['encoder_direction_valid']), 'seconds': time.perf_counter()-start,
            'peak_MiB': torch.cuda.max_memory_allocated()//2**20 if device.type == 'cuda' else 0}), flush=True)
        optimizer.step()
        projection_updated = True
        del output, loss, parts, image, target, skeleton
    for hook in hooks:
        hook.remove()
    model.zero_grad(set_to_none=True)
    model.eval()
    model.swin_unet.global_topology.raw_alpha.data.fill_(.1)
    seen = []
    hook = model.swin_unet.global_topology.node_projection.register_forward_hook(lambda *args: seen.append(True))
    with torch.no_grad():
        fixed = torch.randn(1, 3, 512, 512, device=device)
        before = model(fixed)[0].clone()
        empty = model(torch.zeros_like(fixed))
        zeros = torch.zeros_like(empty[0])
        empty_loss, parts = criterion(empty[0], surface_gt=zeros, skeleton_gt=zeros,
            skeleton_dilate_gt=zeros, stage_outputs=empty[4], encoder_aux_outputs=empty[5],
            boundary_logits=empty[1], skeleton_logits=empty[2], connectivity_logits=empty[3])
        assert torch.isfinite(empty_loss) and float(parts['encoder_direction_valid']) == 0
    hook.remove()
    assert seen, 'Global topology not executed'
    del empty, empty_loss, parts, zeros
    # Strict roundtrip on CPU so the second full model does not compete for GPU RAM.
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = make_model(True)
    restored.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True), strict=True)
    buffer.close()
    for name, value in restored.state_dict().items():
        assert torch.equal(value, model.state_dict()[name].cpu()), name
    del restored
    print('STRICT_ROUNDTRIP_EMPTY_GT_GLOBAL_OK', flush=True)
    if native:
        # Actual 25-window model execution, then compare one-window output in the blend.
        start = time.perf_counter()
        result, _ = stitch_logits(model, torch.zeros(1, 3, 1500, 1500, device=device))
        assert result.shape == (1, 1, 1500, 1500) and torch.isfinite(result).all()
        print('NATIVE1500_25_WINDOWS_OK seconds=' + str(time.perf_counter()-start), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--module_only', action='store_true')
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(4)
    module_checks(device)
    if not args.module_only:
        verify_persistent_worker_epoch_sync()
        full_checks(device, args.batch_sizes, args.native)


if __name__ == '__main__':
    main()
