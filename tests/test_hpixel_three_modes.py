from types import SimpleNamespace

import numpy as np
import pytest
import torch

from networks.h_pixel_reassembly import install_reassembly_from_args
from networks.native_h0_relations import install_native_relations
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from scripts.hpixel_diagnostic_utils import (PixelProbe, difficult_background, distribution,
                                             local_closed, owner_mask, state_digest)
from scripts.diagnose_hpixel_three_modes import selected_threshold, summarize_comparison, topology_manifest


@pytest.fixture(params=['pixel', 'no_a', 'encoder_h'])
def core(request):
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    value = SwinTransformerSys(img_size=128, window_size=8, num_classes=1, return_skeleton=True,
        structure_profile='stage23_boundary_0626', enable_highres_structure_stream=True,
        enable_h3_surface_fusion=True, enable_global_topology=True, stage_skeleton_mode='direct',
        remove_stage2_pre_topology_source=True)
    wrapper = SimpleNamespace(swin_unet=value)
    install_native_relations(wrapper)
    install_reassembly_from_args(wrapper, dict(h_pixel_reassembly=True,
        no_encoder_a1a2=request.param != 'pixel', h_encoder_exchange=request.param == 'encoder_h'))
    value.eval().requires_grad_(False)
    with torch.no_grad():
        value.h_pixel_reassembly.out.weight.normal_(std=.03)
        value.h_pixel_reassembly.out.bias.normal_(mean=.2, std=.1)
    return value


def test_full_model_interventions_bias_replay_and_exact_restoration(core):
    before = state_digest(core)
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        original = core(image)[0].clone()
        with PixelProbe(core) as probe:
            probe.set_mode('normal')
            normal = core(image)
            assert torch.equal(normal[0], original)
            probe.set_mode('c_prior_one')
            uniform = core(image)
            assert probe.upstream_checks == 1
            assert not np.allclose(probe.records['normal']['weights'], probe.records['c_prior_one']['weights'])
            assert probe.records['c_prior_one']['residual_change_abs'].max() > 0
            probe.set_mode('residual_off')
            off = core(image)
            bias_only = probe.bias_only_tail(off[4])
            assert not torch.equal(normal[0], off[0])
        assert 'forward' not in core.h_pixel_reassembly.__dict__
        assert not hasattr(core, 'disable_h_pixel_reassembly')
        assert core.h_pixel_reassembly.prior_mode == 'predicted'
        assert torch.equal(core(image)[0], original)
        # Independent complete model evaluation that returns bias only from H reassembly.
        def only_bias(module, args, output):
            return module.out.bias[None, :, None, None].expand_as(output)
        handle = core.h_pixel_reassembly.register_forward_hook(only_bias)
        try:
            expected_bias = core(image)[0]
        finally:
            handle.remove()
        torch.testing.assert_close(bias_only, expected_bias, rtol=0, atol=0)
    assert state_digest(core) == before
    assert not any(p.grad is not None for p in core.parameters())


def test_bias_alone_can_change_output_without_any_spatial_reassembly(core):
    with torch.no_grad():
        core.h_pixel_reassembly.out.weight.zero_()
    with torch.inference_mode(), PixelProbe(core) as probe:
        image = torch.randn(1, 3, 128, 128)
        probe.set_mode('normal')
        normal = core(image)[0]
        probe.set_mode('c_prior_one')
        uniform = core(image)[0]
        probe.set_mode('residual_off')
        off = core(image)
        bias_only = probe.bias_only_tail(off[4])
        assert torch.equal(normal, uniform)
        assert torch.equal(normal, bias_only)
        assert not torch.equal(normal, off[0])
        assert probe.records['normal']['weighted_component_abs'].max() == 0


def test_exception_restores_existing_flags_and_removes_hooks(core):
    core.disable_h_pixel_reassembly = False
    hooks = len(core.surface_h3_fusion._forward_pre_hooks)
    with pytest.raises(RuntimeError, match='deliberate'):
        with PixelProbe(core) as probe:
            probe.set_mode('c_prior_one')
            raise RuntimeError('deliberate')
    assert core.disable_h_pixel_reassembly is False
    assert core.h_pixel_reassembly.prior_mode == 'predicted'
    assert len(core.surface_h3_fusion._forward_pre_hooks) == hooks
    assert not core.h_pixel_reassembly.out._forward_pre_hooks


def test_native_ownership_partitions_overlapping_non_aligned_last_windows():
    from mass_protocol import sliding_positions
    count = np.zeros((1500, 1500), np.uint8)
    for top in sliding_positions(1500, 512, 256):
        for left in sliding_positions(1500, 512, 256):
            count[top:top+512, left:left+512] += owner_mask(top, left)
    assert np.all(count == 1)


def test_local_gap_requires_coverage_and_shared_local_anchor_component():
    mask = np.zeros((40, 40), bool)
    mask[20, 3:37] = True
    gap = dict(pixels=[20*40+x for x in range(16, 24)],
               anchors=[[20*40+x for x in range(8, 12)], [20*40+x for x in range(29, 33)]])
    assert local_closed(mask, gap)
    mask[20, 19:21] = False
    assert not local_closed(mask, gap)
    mask[20] = False
    mask[20, 16:24] = True
    assert not local_closed(mask, gap)


def test_background_proxy_excludes_GT_and_ambiguous_relations():
    road = np.zeros((16, 16), bool)
    road[5, 4:7] = True
    road[6, 9:12] = True
    selected = difficult_background(road)
    assert selected.any()
    assert not (selected & road).any()
    assert selected[4:8, 4:12].any()
    assert not difficult_background(np.zeros((16, 16), bool)).any()
    from datasets.native_relation_targets import native_targets
    real_skeleton = road.copy()
    a = native_targets(road, real_skeleton)
    b = native_targets(road, np.zeros_like(road))
    for key in ('connectivity', 'connectivity_valid', 'ambiguous_cells'):
        assert np.array_equal(a[key], b[key])


def test_threshold_is_fixed_and_evidence_binds_the_exact_checkpoint(tmp_path):
    import json
    evidence = tmp_path/'selection.json'
    evidence.write_text(json.dumps(dict(selected_on='val', checkpoint_sha256='correct', threshold=.46)))
    args = SimpleNamespace(threshold_selection=evidence, threshold=None)
    assert selected_threshold(args, 'correct')[0] == .46
    with pytest.raises(ValueError):
        selected_threshold(args, 'wrong')
    args = SimpleNamespace(threshold_selection=None, threshold=.2)
    assert selected_threshold(args, 'x')[0] == .2
    args.threshold = 0
    with pytest.raises(ValueError):
        selected_threshold(args, 'x')


def test_manifest_single_threshold_no_sweep_and_pinned_topology_protocol(tmp_path):
    manifest = topology_manifest(tmp_path, tmp_path/'out', tmp_path/'model.pth', 'sha', .46)
    assert manifest['protocol']['apls_max_control_nodes'] == 150
    assert manifest['protocol']['topo_seed_density_deg'] == .002
    assert manifest['protocol']['topo_step_deg'] == .0002
    assert manifest['protocol']['tlts_n_paths'] == 200
    assert len(manifest['models']) == 3
    for item in manifest['models']:
        assert len(item['predictions']['val']) == 1
        assert item['predictions']['val'][0]['threshold'] == .46
        assert 'test' not in item['predictions']


def test_distribution_signed_changes_and_empty_regions():
    x = np.array([[-2., 0.], [1., 3.]])
    result = distribution(x, np.ones((2, 2), bool))
    assert result['signed_mean'] == .5
    assert result['abs_mean'] == 1.5
    assert result['increased'] == 2 and result['decreased'] == 1
    empty = distribution(x, np.zeros((2, 2), bool))
    assert empty['pixels'] == 0 and empty['signed_mean'] is None


def test_result_table_reads_real_APLS_column_without_pandas(tmp_path, capsys):
    import csv
    from scripts.show_hpixel_three_modes import show
    output = tmp_path/'topology'
    output.mkdir()
    with (output/'val_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name', 'cldice', 'topo_precision', 'topo_recall',
            'topo_f1', 'apls_bidirectional', 'tlts_infeasible'])
        writer.writeheader()
        writer.writerow(dict(name='normal', cldice=.89, topo_precision=.5, topo_recall=.6,
                             topo_f1=.54, apls_bidirectional=.612345, tlts_infeasible=.35))
    show(tmp_path)
    rendered = capsys.readouterr().out
    assert '0.612345' in rendered
    assert 'APLS' in rendered


def test_native1500_stitching_and_all_report_fields():
    from networks.h_pixel_reassembly import HPixelReassembly
    from scripts.diagnose_hpixel_three_modes import infer_three, analyse_case
    from torch import nn
    from torch.nn import functional as F

    class Head(nn.Module):
        def forward(self, x, z_struct=None):
            # Keep binary masks empty: thinning a dense 1500-square synthetic
            # foreground would dominate a test of window ownership and logits.
            return x[:, :1].tanh()-1.1, None, None, None

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.h_pixel_reassembly = HPixelReassembly(query_channels=2, h_channels=2,
                g_channels=2, latent_channels=2, chunk_rows=128)
            self.surface_h3_fusion = nn.Conv2d(4, 2, 1)
            self.guided_head = Head()
            with torch.no_grad():
                self.h_pixel_reassembly.out.weight.normal_(std=.1)
                self.h_pixel_reassembly.out.bias.fill_(.15)

        def _apply_structure_surface_correction(self, x, h, structures):
            return x

        def forward(self, image):
            x = image[:, :2]
            h = F.avg_pool2d(x, 4)
            g = h*.5
            c = torch.cat([h[:, :1]*(i+1)/8 for i in range(8)], 1)
            structure = F.interpolate(h, scale_factor=4, mode='bilinear', align_corners=False)
            if not getattr(self, 'disable_h_pixel_reassembly', False):
                structure = structure + self.h_pixel_reassembly(x, h, g, c)
            result = self.guided_head(self.surface_h3_fusion(torch.cat((x, structure), 1)), z_struct=h)
            return (*result, [dict(stage=3, connectivity=c)])

    torch.set_num_threads(2)
    torch.manual_seed(8)
    model = Toy().eval().requires_grad_(False)
    gt = torch.zeros(1, 1500, 1500)
    gt[:, 750, 10:1490] = 1
    sample = dict(image=torch.randn(3, 1500, 1500)*.1, mask=gt)
    windows = []
    with torch.inference_mode():
        expected = model(sample['image'][None, :, :512, :512])[0].clone()
        with PixelProbe(model) as probe:
            logits, maps, hard, checked, last_tile, last_normal, native_masks = infer_three(
                model, sample, 'cpu', probe, windows, 'toy', expected, .5)
        assert torch.equal(model(last_tile)[0], last_normal)
    assert checked == 25
    assert len(windows) == 50
    assert all(value.shape == (1500, 1500) for value in logits.values())
    assert np.allclose(maps['normal']['weights'].sum(0), 1)
    gap = dict(id=7, length=4, length_bin='3-7', pixels=[750*1500+x for x in range(747, 751)],
               anchors=[[750*1500+740], [750*1500+759]])
    entry = dict(shape=[1500, 1500], gaps=[gap])
    pred, rows, fixed, changes, weights, residual = analyse_case(
        'toy', gt[0].numpy() > .5, logits, maps, hard, entry, .5, .15, native_masks)
    assert len(rows) == 3 and len(fixed) == 3
    assert {r['gap_id'] for r in fixed} == {7}
    assert len(changes) == 20 and len(weights) == 36 and len(residual) == 8
    summary = summarize_comparison(rows)
    assert len(summary) == 3
    assert summary[0]['added_fp'] == 0 and summary[0]['lost_closed_vs_normal'] == 0
    assert any(r['residual_change_from_normal_abs_mean'] > 0 for r in residual if r['mode'] == 'c_prior_one' and r['pixels'])
