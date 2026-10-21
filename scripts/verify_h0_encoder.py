"""Finite, synthetic verification of the H0 encoder exchange; no dataset training."""
import argparse
import io
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from config import get_config
from networks.vision_transformer import SwinUnet
from losses.road_losses import SurfaceStructureLoss, build_soft_stage_skeleton_target


def verify_persistent_worker_epoch_sync():
    import cv2
    import numpy as np
    from torch.utils.data import DataLoader
    from datasets.dataset_road_skeleton import RoadSkeletonDataset

    with tempfile.TemporaryDirectory(prefix='h0_epoch_sync_') as temp_dir:
        root = Path(temp_dir)
        image_dir = root / 'train' / 'image'
        mask_dir = root / 'train' / 'mask'
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        y, x = np.mgrid[:100, :100]
        rgb = np.stack((x, y, (x + y) % 256), axis=-1).astype(np.uint8)
        cv2.imwrite(str(image_dir / 'sample.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_dir / 'sample.png'), np.zeros((100, 100), dtype=np.uint8))
        dataset = RoadSkeletonDataset(
            root, split='train', image_size=32, source_patch_size=100,
            random_crop_train=True, random_crops_per_image=1,
            random_crop_seed=1234,
        )
        loader = DataLoader(dataset, batch_size=1, num_workers=1,
                            persistent_workers=True, shuffle=False)
        dataset.set_epoch(0)
        first = next(iter(loader))
        first_position = (int(first['tile_top'][0]), int(first['tile_left'][0]))
        dataset.set_epoch(1)
        second = next(iter(loader))
        second_position = (int(second['tile_top'][0]), int(second['tile_left'][0]))
        del loader
        assert first_position != second_position, (first_position, second_position)
        print('PERSISTENT_WORKER_EPOCH_SYNC_OK', first_position, '->',
              second_position, flush=True)


def make_model(exchange):
    args = SimpleNamespace(
        cfg=str(ROOT / 'configs/swin_tiny_patch4_window7_224_lite.yaml'),
        opts=None, batch_size=2, img_size=512, zip=False, cache_mode='',
        resume='', accumulation_steps=1, use_checkpoint=False,
        amp_opt_level='', tag='', eval=False, throughput=False,
    )
    return SwinUnet(
        get_config(args), img_size=512, num_classes=1, return_skeleton=True,
        structure_profile='stage23_boundary_0626',
        enable_highres_structure_stream=True, highres_structure_channels=64,
        highres_structure_fuse_stages='stage23',
        highres_structure_fusion_mode='stage23',
        enable_global_topology=True, global_topology_max_nodes=32,
        global_topology_heads=4, global_topology_alpha_max=0.05,
        enable_h3_surface_fusion=True, remove_stage2_pre_topology_source=True,
        stage_skeleton_mode='direct', encoder_structure_exchange=exchange,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 2])
    args = parser.parse_args()
    verify_persistent_worker_epoch_sync()
    device = torch.device(args.device)
    torch.manual_seed(1234)
    model = make_model(True).to(device).eval()
    torch.manual_seed(1234)
    control = make_model(False)
    for key, value in control.state_dict().items():
        assert torch.equal(value, model.state_dict()[key].cpu()), key
    print('SHARED_INITIALIZATION_OK tensors=', len(control.state_dict()), flush=True)
    del control
    sequence = []
    hooks = []
    stem_calls = {'stem': 0, 'adapter': 0}
    decoder_h = []
    global_h = []

    def named(name):
        def hook(*_):
            sequence.append(name)
        return hook

    for stage in range(3):
        layer = model.swin_unet.layers[stage]
        hooks.append(layer.blocks[-1].register_forward_hook(named(f'swin{stage + 1}')))
        hooks.append(model.swin_unet.encoder_exchanges[stage].register_forward_hook(
            named(f'exchange{stage + 1}')))
        if stage < 2:
            hooks.append(layer.road_attention_head.register_forward_hook(
                named(f'A{stage + 1}')))
        hooks.append(layer.downsample.register_forward_hook(named(f'merge{stage + 1}')))

    def count(which):
        def hook(*_):
            stem_calls[which] += 1
        return hook

    hooks.append(model.swin_unet.prepatch_structure_encoder.register_forward_hook(count('stem')))
    hooks.append(model.swin_unet.highres_skeleton_adapter.register_forward_hook(count('adapter')))
    hooks.append(model.swin_unet.highres_structure_fusion['2'].register_forward_pre_hook(
        lambda _module, inputs: decoder_h.append(inputs[1])))
    global_forward = model.swin_unet.global_topology.forward_feature_anchors
    def tracked_global(*inputs, **kwargs):
        global_h.append(inputs[1])
        return global_forward(*inputs, **kwargs)
    model.swin_unet.global_topology.forward_feature_anchors = tracked_global
    count_params = sum(p.numel() for p in model.parameters())
    for batch in args.batch_sizes:
        stem_calls.update(stem=0, adapter=0)
        sequence.clear()
        decoder_h.clear()
        global_h.clear()
        x = torch.randn(batch, 3, 512, 512, device=device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            result = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        assert result[0].shape == (batch, 1, 512, 512)
        assert len(result[5]) == 3
        for stage, item in enumerate(result[5]):
            scale = 128 // 2 ** stage
            assert item['encoder_road_logits'].shape == (batch, 1, scale, scale)
            assert item['encoder_skeleton_logits'].shape == (batch, 1, 128, 128)
            assert model.swin_unet.last_encoder_features[f'H_E{stage + 1}'].shape == (
                batch, 64, 128, 128)
            assert model.swin_unet.last_encoder_features[f'E{stage + 1}'].shape == (
                batch, 96 * 2 ** stage, scale, scale)
        assert stem_calls == {'stem': 1, 'adapter': 1}, stem_calls
        assert len(decoder_h) == 1 and decoder_h[0] is model.swin_unet.last_encoder_features['H_E3']
        assert len(global_h) == 1 and global_h[0] is model.swin_unet.last_stage_features['H_D3']
        expected = ['swin1', 'exchange1', 'A1', 'merge1',
                    'swin2', 'exchange2', 'A2', 'merge2',
                    'swin3', 'exchange3', 'merge3']
        assert sequence == expected, sequence
        assert torch.isfinite(result[0]).all()
        print(f'BATCH={batch} shape={tuple(result[0].shape)} seconds={elapsed:.3f} '
              f'peak_MiB={torch.cuda.max_memory_allocated(device)//2**20 if device.type == "cuda" else 0}',
              flush=True)
        del result, x

    for hook in hooks:
        hook.remove()
    state = model.state_dict()
    fixed_image = torch.zeros(1, 3, 512, 512, device=device)
    with torch.no_grad():
        before_reload = model(fixed_image)[0]
    packed = io.BytesIO()
    torch.save(state, packed)
    packed.seek(0)
    restored = make_model(True).to(device).eval()
    restored.load_state_dict(torch.load(packed, map_location=device, weights_only=True), strict=True)
    with torch.no_grad():
        after_reload = restored(fixed_image)[0]
    torch.testing.assert_close(before_reload, after_reload, rtol=0, atol=0)
    print('STRICT_CHECKPOINT_AND_INFERENCE_OK', 'parameters=', count_params, flush=True)
    del restored, before_reload, after_reload, fixed_image

    # Encoder-only backward avoids a full decoder training step during verification.
    model.train()
    for task in ('road', 'skeleton'):
        model.zero_grad(set_to_none=True)
        image = torch.randn(1, 3, 512, 512, device=device)
        h0 = model.swin_unet.highres_skeleton_adapter(
            model.swin_unet._build_highres_structure_outputs(image))
        _, _, _, _, aux = model.swin_unet.forward_features(image, h0)
        if task == 'road':
            loss = sum(item['encoder_road_logits'].square().mean() for item in aux) / 3
            head = model.swin_unet.encoder_road_heads[0].weight
        else:
            target = torch.zeros(1, 1, 512, 512, device=device)
            target[:, :, 240:270, 100:400] = 1
            criterion = SurfaceStructureLoss(skeleton_pos_weight=2.0).to(device)
            terms = []
            for item in aux:
                _, soft = build_soft_stage_skeleton_target(
                    target, item['encoder_skeleton_logits'].shape[-2:])
                terms.append(criterion.soft_stage_skeleton_loss(
                    item['encoder_skeleton_logits'], soft))
            loss = torch.stack(terms).mean()
            head = model.swin_unet.encoder_skeleton_heads[0].weight
        loss.backward()
        for name, param in (
            ('head', head),
            ('exchange', model.swin_unet.encoder_exchanges[0].theta_eh),
            ('backbone', model.swin_unet.layers[0].blocks[0].attn.qkv.weight),
        ):
            assert param.grad is not None and torch.isfinite(param.grad).all(), (task, name)
        print(f'{task.upper()}_BACKWARD_OK loss={loss.item():.6f}', flush=True)
        del aux, loss, image

    model.zero_grad(set_to_none=True)
    criterion = SurfaceStructureLoss(
        surface_dice_weight=0.5, skeleton_dice_weight=1.0,
        stage_structure_weights=(0, 0, 0.008, 0.012),
        stage_skeleton_only_loss_factor=0.8,
        stage_skeleton_loss_factor=1.0,
        stage_connectivity_factor=3.0, stage_direction_factor=0.2,
        skeleton_pos_weight=2.0, connectivity_pos_weight=2.0,
        use_masked_connectivity_center_experiment=True,
        connectivity_focal_gamma=1.5, surface_focal_gamma=1.0,
        encoder_road_aux_weight=0.10,
        encoder_skeleton_aux_weight=0.004,
    ).to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    image = torch.randn(1, 3, 512, 512, device=device)
    target = torch.zeros(1, 1, 512, 512, device=device)
    target[:, :, 240:270, 100:400] = 1
    skeleton = torch.zeros_like(target)
    skeleton[:, :, 255:256, 100:400] = 1
    dilated = torch.nn.functional.max_pool2d(skeleton, 3, stride=1, padding=1)
    outputs = model(image)
    loss, parts = criterion(
        outputs[0], surface_gt=target, skeleton_gt=skeleton,
        skeleton_dilate_gt=dilated, stage_outputs=outputs[4],
        encoder_aux_outputs=outputs[5], boundary_logits=outputs[1],
        skeleton_logits=outputs[2], connectivity_logits=outputs[3],
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.swin_unet.encoder_road_heads[0].weight.grad is not None
    assert model.swin_unet.encoder_skeleton_heads[0].weight.grad is not None
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    print('FULL_STEP_OK seconds={:.3f} peak_MiB={} total={:.6f} '
          'encoder_road={:.6f} encoder_skeleton={:.6f}'.format(
              time.perf_counter() - start,
              torch.cuda.max_memory_allocated(device)//2**20 if device.type == 'cuda' else 0,
              loss.item(), parts['encoder_road_weighted'].item(),
              parts['encoder_skeleton_weighted'].item()), flush=True)
    stage2 = [item for item in outputs[4] if item.get('stage') == 2]
    stage3 = [item for item in outputs[4] if item.get('stage') == 3]
    assert len(stage2) == len(stage3) == 1
    assert criterion.stage_structure_weights[2:] == (0.008, 0.012)
    assert criterion.stage_skeleton_only_loss_factor == 0.8
    assert criterion.stage_connectivity_factor == 3.0
    assert criterion.stage_direction_factor == 0.2
    assert abs(0.008 * 0.8 + 0.012 * 0.8 + 0.004 - 0.020) < 1e-12
    print('STAGE_S_C_D_BUDGET_OK', flush=True)
    model.eval()
    with torch.no_grad():
        empty_outputs = model(torch.zeros_like(image))
        empty_loss, _ = criterion(
            empty_outputs[0], surface_gt=torch.zeros_like(target),
            skeleton_gt=torch.zeros_like(skeleton),
            skeleton_dilate_gt=torch.zeros_like(dilated),
            stage_outputs=empty_outputs[4], encoder_aux_outputs=empty_outputs[5],
            boundary_logits=empty_outputs[1], skeleton_logits=empty_outputs[2],
            connectivity_logits=empty_outputs[3],
        )
        assert torch.isfinite(empty_loss), empty_loss
    print('EMPTY_SKELETON_OK', flush=True)


if __name__ == '__main__':
    main()
