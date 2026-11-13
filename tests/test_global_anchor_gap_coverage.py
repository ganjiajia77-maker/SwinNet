import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from diagnostics.global_anchor_gap_geometry import (distinct_pair, gap_geometry,
    local_status, membership, summarize_gap, window_hits)
from h0_phase_training import SAMPLING_PROTOCOL
from networks.keypoint_global_topology import KeypointGuidedGlobalTopology
from scripts import diagnose_h0_global_anchor_gaps as diagnostic


def toy_gap(size=128, y=60, left=48, right=80):
    gt = np.zeros((size, size), bool)
    gt[y, 4:size-4] = True
    gap = dict(id=7, length=right-left-7,
        pixels=[y*size+x for x in range(left+4, right-3)],
        anchors=[[y*size+left], [y*size+right]], length_bin='16-31')
    return gt, gap


def record(candidates, selected, top=0, left=0, tile=128):
    return dict(top=top, left=left, tile=tile, alpha_global=.02,
        candidate_coords=np.asarray(candidates, np.int64).reshape(-1, 2),
        candidate_valid=np.ones(len(candidates), bool), candidate_scores=np.ones(len(candidates), np.float32),
        selected_coords=np.asarray(selected, np.int64).reshape(-1, 2),
        selected_valid=np.ones(len(selected), bool), selected_scores=np.ones(len(selected), np.float32),
        selected_candidate_indices=np.arange(len(selected), dtype=np.int64))


def test_capture_is_actual_topk_FPS_and_does_not_change_outputs():
    module = KeypointGuidedGlobalTopology(8, 4, max_nodes=4, heads=2, enabled=True)
    score = torch.ones(2, 1, 20, 20)
    score[1] = 0
    score[1, 0, 1, 1] = 2
    ordinary = module._extract_fps_anchors(score)
    module.capture_anchor_selection = True
    observed = module._extract_fps_anchors(score)
    assert all(torch.equal(a, b) for a, b in zip(ordinary[:3], observed[:3]))
    trace = module.last_anchor_selection
    assert trace['candidate_coords'].shape == (2, 256, 2)
    assert trace['candidate_valid'][1].sum() == 1
    for batch in (0, 1):
        valid = trace['selected_valid'][batch]
        indices = trace['selected_candidate_indices'][batch, valid]
        assert torch.equal(trace['candidate_coords'][batch, indices], trace['selected_coords'][batch, valid])
        assert torch.equal(trace['candidate_scores'][batch, indices], trace['selected_scores'][batch, valid])
        assert torch.all(trace['selected_candidate_indices'][batch, ~valid] == -1)


def test_zero_score_has_no_valid_nodes_or_false_origin_hit():
    module = KeypointGuidedGlobalTopology(8, 4, max_nodes=4, heads=2, enabled=True)
    module.capture_anchor_selection = True
    _, valid, _, _ = module._extract_fps_anchors(torch.zeros(1, 1, 8, 8))
    assert not valid.any()
    assert not module.last_anchor_selection['candidate_valid'].any()


def test_correct_two_side_pair_and_candidate_vs_FPS_loss():
    gt, gap = toy_gap()
    geometry = gap_geometry(gt, gap)
    assert geometry['valid']
    nodes = [[60, 48], [60, 80]]
    result, _ = summarize_gap([record(nodes, nodes)], geometry)
    assert result['selected_correct_pair'] and result['candidates_both_correct']
    assert result['windows_seeing_both_GT_sides'] == 1
    lost, _ = summarize_gap([record(nodes, nodes[:1])], geometry)
    assert lost['candidates_both_correct'] and not lost['selected_correct_pair']
    assert not lost['selected_both_geometric']


def test_nearby_parallel_GT_road_hits_geometry_but_is_not_the_correct_road():
    gt, gap = toy_gap()
    gt[68, 4:124] = True
    geometry = gap_geometry(gt, gap, tolerance=12)
    assert geometry['valid']
    result, _ = summarize_gap([record([[68, 48], [68, 80]], [[68, 48], [68, 80]])], geometry)
    assert result['candidates_both_geometric'] and result['selected_both_geometric']
    assert not result['candidates_both_correct'] and not result['selected_correct_pair']
    assert result['selected_GT_rejected_occurrences'] == 2


def test_cross_window_union_does_not_create_a_pair():
    gt, gap = toy_gap(size=64, y=32, left=18, right=44)
    geometry = gap_geometry(gt, gap, tolerance=4)
    assert geometry['valid']
    records = [record([[16, 18]], [[16, 18]], top=16, tile=32),
               record([[16, 12]], [[16, 12]], top=16, left=32, tile=32)]
    result, _ = summarize_gap(records, geometry)
    assert result['both_correct_only_across_windows']
    assert not result['selected_correct_pair'] and result['no_joint_window']


def test_background_and_image_boundary_do_not_wrap_into_GT_sides():
    gt, gap = toy_gap()
    geometry = gap_geometry(gt, gap)
    result = membership(np.array([[-1, -1], [1000, 48], [60, 48], [61, 80]]), geometry, 'strict')
    assert not result[:2].any() and result[2, 0] and not result[3].any()


def test_multiple_anchor_groups_are_unknown_not_arbitrarily_paired():
    gt, gap = toy_gap()
    gap['anchors'].append([60*128+100])
    geometry = gap_geometry(gt, gap)
    assert not geometry['valid'] and geometry['reason'] == 'not_two_anchor_groups'
    result, _ = summarize_gap([record([[60, 48], [60, 80]], [[60, 48], [60, 80]])], geometry)
    assert result['selected_correct_pair'] is None and result['candidates_both_geometric'] is None


def test_invalid_GT_reference_is_excluded():
    gt, gap = toy_gap()
    gt[60, 60] = False
    geometry = gap_geometry(gt, gap)
    assert not geometry['valid'] and geometry['reason'] == 'fixed_reference_pixels_not_GT_road'


def test_anchor_connection_is_separate_from_legacy_coverage_proxy():
    gt, gap = toy_gap()
    pred = gt.copy()
    pred[61, 4:124] = True
    pred[60, 52:77] = False
    status = local_status(pred, gap)
    assert status['anchor_connected_local'] and not status['still_disconnected']
    assert status['gap_coverage'] == 0 and not status['closed_proxy']


def test_one_node_cannot_be_both_endpoints():
    assert distinct_pair(np.array([[True, True]])) is None
    assert distinct_pair(np.array([[True, False], [False, True]])) == (0, 1)


def test_single_node_in_overlapping_tolerance_disks_is_not_two_sides():
    gt, gap = toy_gap(left=52, right=68)
    geometry = gap_geometry(gt, gap, tolerance=12)
    result, _ = summarize_gap([record([[60, 60]], [[60, 60]])], geometry)
    assert not result['candidates_both_geometric']
    assert not result['selected_both_geometric'] and not result['selected_correct_pair']


def test_real_phase_forward_capture_restores_methods_and_every_buffer():
    from test_encoder_h_postmerge_phase import make_core
    from networks.encoder_h_postmerge_phase import install_encoder_h_postmerge_phase
    core = make_core().eval()
    model = nn.Module(); model.swin_unet = core
    model.forward = lambda image: core(image)
    install_encoder_h_postmerge_phase(model)
    model.requires_grad_(False)
    before = diagnostic.state_digest(model)
    method = core.global_topology._extract_fps_anchors.__func__
    probability, records, same = diagnostic.capture_image(model, torch.randn(3, 128, 128),
        torch.device('cpu'), check_first=True, tile=128, stride=64)
    assert same and probability.shape == (128, 128) and len(records) == 1
    assert records[0]['candidate_coords'].shape == (256, 2)
    assert records[0]['selected_coords'].shape == (32, 2)
    assert core.global_topology._extract_fps_anchors.__func__ is method
    assert 'capture_anchor_selection' not in core.global_topology.__dict__
    assert 'last_anchor_selection' not in core.global_topology.__dict__
    assert diagnostic.state_digest(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_cohort_uses_unique_gaps_and_excludes_uncertain_side_labels():
    rows = []
    for valid, closed in ((True, False), (True, True), (False, False)):
        rows.append(dict(side_GT_valid=valid, candidates_both_geometric=True,
            selected_both_geometric=True, candidates_both_correct=True if valid else None,
            selected_correct_pair=closed if valid else None, both_correct_only_across_windows=False,
            no_joint_window=False))
    summary = diagnostic.aggregate(rows, 'all')
    assert summary['gaps'] == 3 and summary['valid_two_side_gaps'] == 2
    assert summary['selected_correct_pair'] == 1 and summary['selected_correct_pair_rate'] == .5
    assert summary['correct_candidates_but_no_selected_pair'] == 1


def test_complete_198_gap_report_reuses_cached_masks_and_threshold(tmp_path, monkeypatch):
    root = tmp_path/'data'
    evaluation = tmp_path/'evaluation'
    baseline = tmp_path/'baseline'
    output = tmp_path/'diagnostic'
    for folder in ('val_probabilities', 'val_masks'):
        (evaluation/folder).mkdir(parents=True)
    (baseline/'val_probabilities').mkdir(parents=True)
    checkpoint = tmp_path/'best.pth'; checkpoint.write_bytes(b'fixed checkpoint')
    sha = diagnostic.file_digest(checkpoint)
    names = [f'case{i}' for i in range(14)]
    gt, example = toy_gap(size=1500)
    probability = np.zeros((1500, 1500), np.float32)
    images = {}
    for index, case in enumerate(names):
        gaps = [dict(example, id=j) for j in range(16 if index == 0 else 14)]
        images[case] = dict(shape=[1500, 1500], gaps=gaps)
        np.save(evaluation/'val_probabilities'/(case+'.npy'), probability)
        np.save(baseline/'val_probabilities'/(case+'.npy'), gt.astype(np.float32))
        cv2.imwrite(str(evaluation/'val_masks'/(case+'.png')), np.zeros_like(gt, np.uint8))
    fixed = tmp_path/'fixed.json'
    fixed.write_text(json.dumps(dict(count=198, p0_reference_threshold=.45, images=images)))
    (evaluation/'val_selection.json').write_text(json.dumps(dict(selected_on='val',
        criterion='global_iou', checkpoint_sha256=sha, threshold=.48)))
    (evaluation/'protocol.json').write_text(json.dumps(dict(checkpoint_sha256=sha,
        window=512, stride=256, ramp_min=.1, data_root=str(root.resolve()))))
    (baseline/'val_selection.json').write_text(json.dumps(dict(selected_on='val',
        criterion='global_iou', checkpoint_sha256='baseline', threshold=.45)))
    (baseline/'protocol.json').write_text(json.dumps(dict(checkpoint_sha256='baseline',
        window=512, stride=256, ramp_min=.1, data_root=str(root.resolve()))))
    class Data:
        def __init__(self, *args, **kwargs):self.image_files = [case+'.png' for case in names]
        def __len__(self):return 14
        def __getitem__(self, index):
            return dict(case_name=names[index], image=torch.zeros(1, 1500, 1500).expand(3, -1, -1),
                        mask=torch.from_numpy(gt[None].astype(np.float32)))
    model = nn.Module(); model.swin_unet = nn.Module()
    model.swin_unet.enable_global_topology = True
    model.swin_unet.global_topology = nn.Module()
    model.swin_unet.global_topology.max_nodes = 32
    model.register_buffer('unchanged', torch.ones(1))
    saved = dict(sampling_protocol=SAMPLING_PROTOCOL, img_size=512, amp_dtype='none')
    monkeypatch.setattr(diagnostic, 'RoadSkeletonDataset', Data)
    monkeypatch.setattr(diagnostic, 'load_model', lambda *args: (model, saved, 65))
    records = [record([[60, 48], [60, 80]], [[60, 48], [60, 80]], top=top, left=left, tile=512)
               for top in diagnostic.positions(1500) for left in diagnostic.positions(1500)]
    monkeypatch.setattr(diagnostic, 'capture_image', lambda model, image, device, check_first:
        (probability.copy(), records, True if check_first else None))
    args = SimpleNamespace(checkpoint=checkpoint, evaluation_dir=evaluation, root_path=root,
        fixed_gaps=fixed, output_dir=output, device='cpu', tolerance=12., no_visuals=True,
        baseline_evaluation_dir=baseline)
    diagnostic.run(args)
    summary = json.loads((output/'summary.json').read_text())
    assert summary['complete'] and summary['fixed_gaps'] == 198 and summary['threshold'] == .48
    primary = next(row for row in summary['cohorts'] if row['cohort'] == 'still_disconnected')
    assert primary['gaps'] == 198 and primary['valid_two_side_gaps'] == 198
    assert primary['selected_correct_pair'] == 198
    assert summary['comparison']['threshold'] == .45
    assert summary['comparison']['baseline_closed'] == summary['comparison']['lost'] == 198
    assert summary['comparison']['gained'] == 0
    assert (output/'nodes.csv').is_file() and (output/'gap_windows.csv').is_file()
    assert json.loads((output/'checks.json').read_text())['model_unchanged']


def test_visualization_displays_one_window_with_native_coordinates(tmp_path):
    gt, gap = toy_gap()
    geometry = gap_geometry(gt, gap)
    sample = dict(image=torch.zeros(3, 128, 128), mask=torch.from_numpy(gt[None].astype(np.float32)))
    target = tmp_path/'case.png'
    diagnostic.draw_case(sample, gt, gap, geometry, {},
        [record([[60, 48], [60, 80]], [[60, 48], [60, 80]])], target)
    rendered = cv2.imread(str(target))
    assert rendered is not None and rendered.shape[1] > rendered.shape[0]
