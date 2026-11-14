import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import evaluate_h0_encoder_phase as evaluate


def test_strict_phase_checkpoint_restore_and_EMA_wrong_variant_rejection(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    saved = dict(encoder_h_postmerge_phase=True, global_topology_stage2=True, use_ema=False, img_size=128,
        structure_profile='stage23_boundary_0626', enable_highres_structure_stream=True,
        highres_structure_channels=64, enable_h3_surface_fusion=True, enable_global_topology=True,
        stage_skeleton_mode='direct', remove_stage2_pre_topology_source=True)
    original = evaluate.build_model(saved).eval()
    state = dict(args=saved, epoch=80, model_state_dict=original.state_dict())
    checkpoint = tmp_path/'best.pth';checkpoint.write_bytes(b'checkpoint identity')
    monkeypatch.setattr(evaluate.torch, 'load', lambda *a, **k: state)
    state['model_state_dict']['swin_unet.layers.2.downsample.road_alpha'] = torch.tensor(.1)
    restored, args, epoch = evaluate.load_model(checkpoint, torch.device('cpu'))
    assert epoch == 80 and not restored.training
    assert not any(p.requires_grad for p in restored.parameters())
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        assert torch.equal(original(image)[0], restored(image)[0])
    state['model_state_dict'] = dict(state['model_state_dict'])
    state['model_state_dict'].pop(next(iter(state['model_state_dict'])))
    with pytest.raises(RuntimeError, match='Missing key'):
        evaluate.load_model(checkpoint, torch.device('cpu'))
    with pytest.raises(ValueError, match='EMA-off'):
        evaluate.build_model(dict(saved, use_ema=True))
    with pytest.raises(ValueError, match='Stage2-global'):
        evaluate.build_model(dict(saved, global_topology_stage2=False))


def test_native1500_512_256_and_original_point_one_weighting():
    torch.set_num_threads(2)
    class Constant(torch.nn.Module):
        def __init__(self):super().__init__();self.starts=[]
        def forward(self, tile):
            self.starts.append(int(tile[0, 0, 0, 0]))
            assert tile.shape == (1, 3, 512, 512)
            return torch.zeros(1, 1, 512, 512),
    value = torch.arange(1500*1500, dtype=torch.float32).reshape(1, 1500, 1500)
    sample = dict(image=value.expand(3, -1, -1), mask=torch.zeros(1, 1500, 1500))
    model = Constant().eval()
    probability, gt = evaluate.infer(model, sample, torch.device('cpu'))
    assert np.all(probability == .5) and not gt.any()
    assert evaluate.positions(1500) == [0, 256, 512, 768, 988]
    assert model.starts == [y*1500+x for y in evaluate.positions(1500) for x in evaluate.positions(1500)]
    weight = evaluate.blend_weight('cpu')
    assert weight.shape == (1, 1, 512, 512)
    assert float(weight[0, 0, 0, 0]) == pytest.approx(.01)


def test_pipeline_selects_on_val_then_uses_exact_fixed_threshold_on_test(tmp_path, monkeypatch):
    # Tiny synthetic masks isolate pipeline control flow; real native shapes
    # and overlap weighting are checked separately above.
    class Dataset:
        def __init__(self, root, split, **kwargs):
            self.split = split
            assert kwargs == dict(image_size=None, source_patch_size=1500, augment=False, return_full_image=True)
        def __len__(self):return 14 if self.split == 'val' else 49
        def __getitem__(self, index):return dict(case_name=f'{self.split}_{index:03d}', split=self.split)
    calls = []
    def inference(model, sample, device):
        calls.append(sample['split'])
        probability = np.array([[.7, .4], [.7, .4]], dtype=np.float32)
        gt = np.array([[True, False], [True, False]]) if sample['split'] == 'val' else np.zeros((2, 2), bool)
        return probability, gt
    monkeypatch.setattr(evaluate, 'RoadSkeletonDataset', Dataset)
    monkeypatch.setattr(evaluate, 'infer', inference)
    monkeypatch.setattr(evaluate, 'THRESHOLDS', np.array([.2, .6, .8], np.float32))
    monkeypatch.setattr(evaluate, 'load_model', lambda *a: (object(), dict(img_size=512, amp_dtype='none'), 78))
    checkpoint = tmp_path/'best.pth';checkpoint.write_bytes(b'model')
    out = tmp_path/'evaluation'
    args = SimpleNamespace(checkpoint=checkpoint, output_dir=out, device='cpu', root_path=tmp_path/'data')
    evaluate.inference(args)
    selected = json.loads((out/'val_selection.json').read_text())
    assert selected['selected_on'] == 'val' and selected['criterion'] == 'global_iou'
    assert selected['threshold'] == float(np.float32(.6))
    assert calls == ['val']*14+['test']*49
    assert len(list((out/'val_masks').glob('*.png'))) == 14
    assert len(list((out/'test_masks').glob('*.png'))) == 49
    with (out/'test_per_image.csv').open(newline='') as stream:rows = list(csv.DictReader(stream))
    assert all(float(row['threshold']) == selected['threshold'] and int(row['fp']) == 2 for row in rows)
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['protocol']['apls_max_control_nodes'] == 150
    assert manifest['protocol']['topo_seed_density_deg'] == .002
    assert manifest['protocol']['topo_step_deg'] == .0002
    assert manifest['protocol']['tlts_n_paths'] == 200
    evidence = manifest['models'][0]['validation_selection']['evidence']
    assert Path(evidence) == out/'val_selection.json'


def test_topology_uses_test_entry_no_precreated_directory_and_can_resume(tmp_path, monkeypatch):
    tool = tmp_path/'compare_predictions.py';tool.write_text('# pinned tool placeholder')
    (tmp_path/'inference_complete.json').write_text(json.dumps(dict(complete=True, val_images=14, test_images=49)))
    args = SimpleNamespace(output_dir=tmp_path, topo_tool=tool, workers=4)
    calls = []
    monkeypatch.setattr(evaluate.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    evaluate.topology(args)
    assert calls[0][calls[0].index('--split')+1] == 'test'
    assert '--import_val_selection' in calls[0] and '--resume' not in calls[0]
    assert not (tmp_path/'topology').exists()
    (tmp_path/'topology').mkdir();(tmp_path/'topology'/'run_state.json').write_text('{}')
    evaluate.topology(args)
    assert '--resume' in calls[1]
