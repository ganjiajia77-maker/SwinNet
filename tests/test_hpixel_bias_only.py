import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from networks.h_pixel_reassembly import install_reassembly_from_args
from networks.native_h0_relations import install_native_relations
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from scripts.diagnose_hpixel_bias_only import BiasOnly, case_metrics, merge_topology, overlap_rows
from scripts.hpixel_diagnostic_utils import PixelProbe, state_digest


@pytest.mark.parametrize('variant', ['pixel', 'no_a1a2', 'encoder_h'])
def test_bias_only_matches_existing_tail_replay_skips_QKV_and_restores_normal(variant):
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    core = SwinTransformerSys(img_size=128, window_size=8, num_classes=1, return_skeleton=True,
        structure_profile='stage23_boundary_0626', enable_highres_structure_stream=True,
        enable_h3_surface_fusion=True, enable_global_topology=True, stage_skeleton_mode='direct',
        remove_stage2_pre_topology_source=True)
    wrapper = SimpleNamespace(swin_unet=core)
    install_native_relations(wrapper)
    install_reassembly_from_args(wrapper, dict(h_pixel_reassembly=True,
        no_encoder_a1a2=variant != 'pixel', h_encoder_exchange=variant == 'encoder_h'))
    core.eval().requires_grad_(False)
    with torch.no_grad():
        core.h_pixel_reassembly.out.weight.normal_(std=.03)
        core.h_pixel_reassembly.out.bias.normal_(mean=.2, std=.1)
    before = state_digest(core)
    with torch.inference_mode():
        image = torch.randn(1, 3, 128, 128)
        normal = core(image)[0].clone()
        with PixelProbe(core) as probe:
            probe.set_mode('normal'); core(image)
            probe.set_mode('residual_off'); result = core(image)
            expected = probe.bias_only_tail(result[4])
        def forbidden(*unused):
            raise AssertionError('Bias-only must not run query/key/value matching')
        handles = [getattr(core.h_pixel_reassembly, key).register_forward_hook(forbidden)
                   for key in ('query', 'key', 'value')]
        try:
            with BiasOnly(core):
                actual = core(image)[0]
        finally:
            for handle in handles:handle.remove()
        assert torch.equal(actual, expected)
        assert torch.equal(core(image)[0], normal)
    assert 'forward' not in core.h_pixel_reassembly.__dict__
    assert not hasattr(core, 'disable_h_pixel_reassembly')
    assert state_digest(core) == before


def test_bias_only_exception_restores_preexisting_attributes():
    module = torch.nn.Module()
    module.out = torch.nn.Conv2d(1, 1, 1)
    module.forward = lambda *args: 'original'
    core = SimpleNamespace(h_pixel_reassembly=module, disable_h_pixel_reassembly=True)
    old = module.forward
    with pytest.raises(RuntimeError):
        with BiasOnly(SimpleNamespace(swin_unet=core)):
            assert core.disable_h_pixel_reassembly is False
            raise RuntimeError('deliberate')
    assert module.forward is old and core.disable_h_pixel_reassembly is True


def test_bias_masks_get_actual_segmentation_and_frozen_gap_metrics():
    gt = np.zeros((40, 40), bool);gt[20, 3:37] = True
    normal = gt.copy();normal[20, 16:24] = False
    entry = dict(gaps=[dict(id=7, length=8, length_bin='8-15',
        pixels=[20*40+x for x in range(16, 24)],
        anchors=[[20*40+10], [20*40+30]])])
    row, fixed = case_metrics('toy', gt, normal, gt, entry, .45)
    assert row['mode'] == 'bias_only' and row['closed'] == 1
    assert row['newly_closed_vs_normal'] == 1 and row['lost_closed_vs_normal'] == 0
    assert row['added_tp'] == 8 and row['added_fp'] == 0
    assert row['iou'] == 1 and row['cldice'] == 1
    assert fixed[0]['gap_id'] == 7 and fixed[0]['mode_closed']


def test_equal_closure_counts_do_not_imply_identical_gap_IDs():
    fixed = [dict(case='toy', gap_id=1, mode='normal', mode_closed=True),
             dict(case='toy', gap_id=2, mode='bias_only', mode_closed=True)]
    row = next(row for row in overlap_rows(fixed) if row['mode'] == 'bias_only')
    assert row['normal_closed'] == row['mode_closed'] == 1
    assert row['common_closed'] == 0 and row['normal_only'] == row['mode_only'] == 1


def topology_fixture(tmp_path):
    source = tmp_path/'source';out = tmp_path/'bias'
    (source/'topology').mkdir(parents=True);(out/'topology_fixed_val').mkdir(parents=True)
    protocol = dict(apls_max_control_nodes=150, topo_seed_density_deg=.002, topo_step_deg=.0002)
    masks = dict(gt='gt', normal='n', c_prior_one='c', residual_off='o')
    fields = dict(iou=.6, cldice=.88, topo_precision=.5, topo_recall=.6, topo_f1=.54,
                  apls_bidirectional=.61, tlts_infeasible=.3)
    common = dict(protocol=protocol, metric_implementation_sha256={'tool':'same'}, case_ids_sha256='cases')
    old = dict(**common, binary_content_sha256=masks,
               summaries=[dict(name=name, **fields) for name in ('normal', 'c_prior_one', 'residual_off')])
    new = dict(**common, binary_content_sha256=dict(gt='gt', bias_only='b'),
               summaries=[dict(name='bias_only', **fields)])
    for path, value in [(source/'topology'/'val_report.json', old),
                        (out/'topology_fixed_val'/'val_report.json', new),
                        (out/'checks.json', dict(cached_binary_content_sha256=masks))]:
        path.write_text(json.dumps(value))
    return source, out


def test_merge_reuses_old_metrics_and_checks_protocol_implementation_and_binary_inputs(tmp_path):
    source, out = topology_fixture(tmp_path)
    assert merge_topology(source, out)
    import csv
    with (out/'topology_comparison.csv').open(newline='') as stream:rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and rows[-1]['mode'] == 'bias_only'
    assert float(rows[-1]['apls_bidirectional']) == .61
    assert not json.loads((out/'topology_merge.json').read_text())['old_modes_recomputed']
    old_path = source/'topology'/'val_report.json'
    old = json.loads(old_path.read_text())
    old['protocol']['apls_max_control_nodes'] = 500
    old_path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match='protocol'):merge_topology(source, out)


def test_merge_rejects_changed_cached_mask_content(tmp_path):
    source, out = topology_fixture(tmp_path)
    path = out/'checks.json';value = json.loads(path.read_text())
    value['cached_binary_content_sha256']['normal'] = 'changed'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='Cached masks'):merge_topology(source, out)
