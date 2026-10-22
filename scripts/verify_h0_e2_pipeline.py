"""Check production argument/loss/checkpoint paths and synthetic val/test exports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from h0_encoder_experiment import train_command, main as experiment_main
from h0_e2_config import AGGREGATION_CONFIG, direction_config, validate_checkpoint, inherit_checkpoint_args
from h0_encoder_protocol import MODEL_NAME, BASE_COMMIT, DIRECT_BASE_COMMIT, validate_training_args
from losses.encoder_direction_loss import DEFAULT_DIRECTION_CONFIG
from scripts.verify_h0_encoder import make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--pretrain_ckpt', type=Path, required=True)
    options = parser.parse_args()
    torch.set_num_threads(4)
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(root_path=output/'data', output_dir=output, run_name='synthetic',
                           pretrain_ckpt=options.pretrain_ckpt.resolve(), num_workers=0, seed=1234,
                           **DEFAULT_DIRECTION_CONFIG)
    command = train_command(args)
    # Import the actual entry to validate argument parsing and the actual criterion;
    # the guarded training main is not executed.
    sys.argv = command[2:]
    import train_image as entry
    entry.args.num_classes = 1
    validate_training_args(entry.args)
    criterion = entry.build_criterion(entry.args, entry.get_final_loss_weights(entry.args), torch.device('cpu'))
    assert criterion.stage_direction_factor == 0
    assert abs(.008*.8 + .012*.8 + .004 - .020) < 1e-12
    model = make_model(True)
    config = model.config
    config.defrost()
    config.MODEL.PRETRAIN_CKPT = str(options.pretrain_ckpt)
    config.freeze()
    loaded = model.load_from(config)
    assert len(loaded) == 225
    assert not any('e2_direction' in n for n in loaded)
    new_names = [n for n, _ in model.named_parameters() if n not in loaded]
    assert any('e2_direction.gate_head' in n for n in new_names)
    checkpoint = dict(architecture_id=MODEL_NAME, base_commit=BASE_COMMIT,
        direct_base_commit=DIRECT_BASE_COMMIT, args=vars(entry.args),
        encoder_direction_config=direction_config(entry.args),
        encoder_aggregation_config=AGGREGATION_CONFIG,
        model_state_dict=model.state_dict(), ema_state_dict=None, epoch=0)
    validate_checkpoint(checkpoint)
    sys.argv = ['verify']
    restored = SimpleNamespace(**vars(entry.args))
    restored.pca_radius = 99
    inherit_checkpoint_args(restored, checkpoint, training=True)
    assert restored.pca_radius == 5
    sys.argv = ['verify', '--pca_radius', '99']
    restored.pca_radius = 99
    try:
        inherit_checkpoint_args(restored, checkpoint, training=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Explicit conflicting architecture override accepted')
    sys.argv = ['verify']
    ckpt_path = output/'synthetic.pth'
    torch.save(checkpoint, ckpt_path)
    del model, checkpoint, criterion
    # A native1500 road sample in each split, exercising actual CLI + unified metrics.
    image = np.full((1500, 1500, 3), 120, np.uint8)
    mask = np.zeros((1500, 1500), np.uint8)
    mask[740:760, 50:1450] = 255
    for split in ('val', 'test'):
        for kind, array in [('image', image), ('mask', mask)]:
            folder = args.root_path/split/kind
            folder.mkdir(parents=True)
            assert cv2.imwrite(str(folder/'synthetic.png'), array)
    env = dict(os.environ, OPENCV_LOG_LEVEL='ERROR', OMP_NUM_THREADS='4')
    report = output/'unified'
    for mode in ('sweep', 'test', 'topology'):
        command = [sys.executable, '-u', str(ROOT/'h0_encoder_experiment.py'), mode,
                   '--root_path', str(args.root_path), '--model_path', str(ckpt_path),
                   '--output_dir', str(report), '--gpu', '0']
        if mode == 'sweep':
            command += ['--thresholds', '.45', '.5']
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    print('PRODUCTION_ARGS_LOSS_225_PRETRAIN_STRICT_CONFIG_AND_EVAL_PIPELINE_OK', flush=True)


if __name__ == '__main__':
    main()
