"""Actual H0 shape/gradient check; --size 512 checks the production grid."""
import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import _C
from networks.vision_transformer import SwinUnet, TOPOLOGY_ATTENTION_VERSION, load_topology_checkpoint_state
from losses.road_losses import SurfaceStructureLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.manual_seed(1234)
    config = _C.clone()
    config.merge_from_file(str(Path(__file__).resolve().parents[1] / 'configs/swin_tiny_patch4_window7_224_lite.yaml'))
    config.DATA.IMG_SIZE = args.size
    model = SwinUnet(config, img_size=args.size, num_classes=1, return_skeleton=True,
                     structure_profile='stage23_boundary_0626', enable_highres_structure_stream=True,
                     enable_h3_surface_fusion=True, stage_skeleton_mode='direct',
                     remove_stage2_pre_topology_source=True).eval()
    state = model.state_dict()
    assert not any('.decoder_structure_blocks.3.connectivity_' in key or '.structure_gate.' in key
                   or '.global_context_head.' in key or '.dca_blocks.0.weight_conv.' in key for key in state)
    load_topology_checkpoint_state(model, state, TOPOLOGY_ATTENTION_VERSION, strict=True)
    try:
        load_topology_checkpoint_state(model, state, 'legacy', strict=True)
        raise AssertionError('Legacy checkpoint accepted')
    except ValueError:
        pass
    core = model.swin_unet
    core.decoder_structure_blocks['2'].local_propagation.beta.data.fill_(.1)
    core.gap_connection.gamma.data.fill_(.1); core.gap_connection.set_epoch(20)
    stage_size = args.size // 4
    p2_inputs, propagation_inputs = [], []
    core.decoder_structure_blocks['2'].p2_head.register_forward_pre_hook(
        lambda module, inputs: p2_inputs.append(inputs[0].detach().clone())
    )
    core.decoder_structure_blocks['2'].local_propagation.register_forward_pre_hook(
        lambda module, inputs: propagation_inputs.append(inputs[0].detach().clone())
    )
    def force_proposals(module, inputs, logits):
        target = torch.full_like(logits, -8)
        y = stage_size//2
        if stage_size == 16:
            target[:, :, y, 3:7] = 8; target[:, :, y, 10:14] = 8
        else:
            target[:, :, y, 20:51] = 8; target[:, :, y, 70:101] = 8
        return logits + (target-logits).detach()
    hook = core.decoder_structure_blocks['2'].p2_head.register_forward_hook(force_proposals)
    image = torch.randn(1, 3, args.size, args.size)
    with torch.set_grad_enabled(args.backward):
        outputs = model(image)
        assert outputs[0].shape == (1, 1, args.size, args.size)
        stages = outputs[-1]
        assert len(p2_inputs) == len(propagation_inputs) == 1
        assert torch.equal(p2_inputs[0], propagation_inputs[0]), 'P2 must read pre-propagation X2'
        local = next(x for x in stages if x.get('stage') == 2)
        gap = next(x for x in stages if x.get('stage') == 'gap_connection')
        assert local['connectivity'].shape == (1, 8, stage_size, stage_size)
        assert len(gap['gap_edges'][0]) == 1
        assert next(x for x in stages if x.get('stage') == 3)['connectivity'] is None
        assert torch.isfinite(outputs[0]).all()
        gradients = {}
        if args.backward:
            gt = torch.zeros(1, 1, args.size, args.size)
            gt[:, :, args.size//2+2, args.size//8:args.size*7//8] = 1
            criterion = SurfaceStructureLoss(skeleton_weight=0., connectivity_weight=0.,
                          stage_structure_weights=(0, 0, .008, .012), stage_connectivity_factor=3.,
                          stage_direction_factor=.2, use_legacy_stage_connectivity_loss=True,
                          use_masked_connectivity_center_experiment=True)
            loss, stats = criterion(outputs[0], surface_gt=gt, skeleton_gt=gt, stage_outputs=stages)
            assert torch.isfinite(loss)
            loss.backward()
            parameters = dict(model.named_parameters())
            for suffix in ('decoder_structure_blocks.2.connectivity_head.edge_linear.weight',
                           'decoder_structure_blocks.2.skeleton_head.out.weight',
                           'decoder_structure_blocks.2.p2_head.weight',
                           'gap_connection.edge_mlp.4.weight'):
                p = parameters['swin_unet.' + suffix]
                assert p.grad is not None and torch.isfinite(p.grad).all()
                gradients[suffix] = float(p.grad.abs().sum())
                assert gradients[suffix] > 0, suffix
            # Separately prove Surface alone reaches both connection predictions.
            model.zero_grad(set_to_none=True)
            surface = model(image)[0]
            surface.square().mean().backward()
            for suffix in ('decoder_structure_blocks.2.connectivity_head.edge_linear.weight',
                           'gap_connection.edge_mlp.4.weight'):
                p = parameters['swin_unet.' + suffix]
                assert p.grad is not None and float(p.grad.abs().sum()) > 0, suffix
    hook.remove()
    print(json.dumps({'input': args.size, 'surface': list(outputs[0].shape),
                      'local_c': list(local['connectivity'].shape), 'gap_edges': 1,
                      'stage3_dense_c': False, 'old_gates': False, 'backward': args.backward,
                      'gradient_l1': gradients}, indent=2))


if __name__ == '__main__':
    main()
