"""Synthetic native1500 end-to-end audit: strict arm loads, 25 windows, all CSVs."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from checks.test_v9_state_gate import model as original_model
import run_v9_multiscale as runner
from diagnostics.audit_v9_multiscale import audit
from networks.v9_multiscale_transport import install


def main():
    torch.set_num_threads(2); torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample = {'case_name': 'synthetic1500', 'image': torch.randn(3, 1500, 1500),
              'mask': torch.zeros(1, 1500, 1500)}
    sample['mask'][:, 748:752, 200:1300] = 1.
    model = original_model().to(device).eval()
    rates = {name: 1e-4 for name in runner.v9.STAGE_MODULES['D']}
    training = {'reference_threshold': .45}
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        root = Path(directory); source = root/'source/stage_D/best.pth'
        runner.v9.save_checkpoint(source, model, {}, 'D', 1, .5)
        with torch.inference_mode():
            probabilities = runner.native_prediction(model, sample, device)
        assert probabilities.shape == (2, 1500, 1500)
        gt = sample['mask'][0].numpy() > .5
        counts = {}
        for field, probability in zip(('p0', 'p1'), probabilities):
            pred = probability >= .45
            counts[field] = {'tp': int((pred & gt).sum()), 'fp': int((pred & ~gt).sum()), 'fn': int((~pred & gt).sum())}
        runner.write_json(source.parent.parent/'direct_val_selection.json', {'checkpoint': str(source.resolve()),
            'identity': {}, 'selected_on': 'val', 'threshold': .45,
            'report': {**counts, 'paired': {'eligible_p0_gaps': 0}}})
        config = {'mode': 'multiscale', 'source_sha256': runner.v9.digest(source), 'identity': {}, 'training': training}
        for mode in ('local', 'multiscale'):
            install(model, mode)
            optimizer = runner.configure(model, rates, 1e-4)
            out = root/mode; out.mkdir()
            arm = dict(config, mode=mode)
            runner.save(out/'best.pth', model, {'identity': {}}, arm, optimizer, 1, .5, {})
            runner.write_json(out/'val_selection.json', {'checkpoint_sha256': runner.v9.digest(out/'best.pth'),
                'config': arm, 'threshold': .45})
        args = SimpleNamespace(output_dir=root, source_checkpoint=source, audit_split='val', device=device,
            seed=1234, min_effect=1e-5, reference_set=None, root_path=root)
        with patch('run_q3_r3_mass.full_dataset', return_value=[sample]):
            audit(args, model, config)
        result = Path((root/'latest_audit_val.txt').read_text().strip())
        for name in ('summary.json', 'fixed_gaps.csv', 'gap_center_sources.csv',
                     'generated_source_reuse_and_bridges.csv', 'fixed_threshold_comparison.csv'):
            assert (result/name).is_file(), name
        summary = json.loads((result/'summary.json').read_text())
        assert summary['reference_set']['images'] == 1
        assert len(summary['metrics']) == 3
        runner.write_json(ROOT/'validation/multiscale_native.json', {'passed': True,
            'image_size': 1500, 'window_size': 512, 'stride': 256, 'windows_per_arm': 25,
            'arms': ['original', 'local', 'multiscale'], 'device': str(device),
            'original_p0_p1_confusion_counts_reproduced': True, 'strict_downstream_loads': True,
            'limits': 'Synthetic frozen provider and synthetic image; no Massachusetts accuracy claim.'})
    print('[NATIVE1500 AUDIT PASSED]', flush=True)


if __name__ == '__main__':
    main()
