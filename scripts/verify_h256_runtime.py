"""Exercise H256 sweep, fixed-threshold test and topology on synthetic native tiles."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffusion_protocol import BASE_COMMIT, EXPERIMENT_BASE, H256_ID, architecture, build_model


def main():
    with tempfile.TemporaryDirectory(dir=ROOT / '.validation') as temporary:
        temporary = Path(temporary)
        data, report = temporary / 'data', temporary / 'report'
        for split in ('val', 'test'):
            (data / split).mkdir(parents=True)
            (data / f'{split}_labels').mkdir()
            image = np.zeros((1500, 1500, 3), dtype=np.uint8)
            label = np.zeros((1500, 1500), dtype=np.uint8)
            cv2.line(label, (400, 750), (1100, 750), 255, 5)
            assert cv2.imwrite(str(data / split / 'example.tiff'), image)
            assert cv2.imwrite(str(data / f'{split}_labels' / 'example.tif'), label)
        arc = architecture(architecture_id=H256_ID)
        torch.manual_seed(1234)
        model = build_model(arc)
        checkpoint = temporary / 'h256.pth'
        torch.save(dict(architecture_id=H256_ID, architecture_config=arc,
                        base_h0_commit=BASE_COMMIT, base_experiment_commit=EXPERIMENT_BASE,
                        model_config=model.config.dump(), model_state_dict=model.state_dict(),
                        args={'use_ema': False, 'seed': 1234}, epoch=1,
                        initialization_manifest={'shared_tensor_sha256': 'synthetic'}), checkpoint)
        del model
        for action in ('sweep', 'test', 'topology'):
            subprocess.run([sys.executable, '-u', str(ROOT / 'h256_experiment.py'), action,
                            '--gpu', '0', '--root_path', str(data), '--output_dir', str(report),
                            '--model_path', str(checkpoint)], check=True)
        selection = json.loads((report / 'val' / 'val_selection.json').read_text())
        chosen = selection['models']['h0_encoder_h256_s2skel_s3diff_semantic_directional']['threshold']
        assert 0.05 <= chosen <= 0.95
        assert (report / 'val' / 'val_threshold_scores.csv').is_file()
        assert (report / 'test' / 'test_comparison.csv').is_file()
        assert cv2.imread(str(report / 'predictions' / 'test' /
                               f'threshold_{chosen:.2f}' / 'surface' / 'example_pred.png'), 0).shape == (1500, 1500)
        summary = {'synthetic_native_size': 1500, 'selected_val_threshold': chosen,
                   'sweep_test_topology': 'passed', 'checkpoint_strict_restore': 'passed'}
    (ROOT / 'H256_RUNTIME_VERIFICATION.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    (ROOT / '.validation').mkdir(exist_ok=True)
    main()
