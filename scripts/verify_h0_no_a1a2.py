"""Compare the ablation with exact 1af9580 code whose A1/A2 paths are disabled."""
import argparse
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from networks import vision_transformer
from verify_h0_encoder import make_model


BASE = '1af958056f5953f0eb6edbffcdb9f1a9db77ba63'
CORE = 'networks/swin_transformer_unet_skip_expand_decoder_sys.py'


def is_removed_key(key):
    return ('road_attention_head.' in key or key.endswith('.road_alpha')
            or '.road_bias_scale_a' in key)


def assert_same_tree(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_same_tree(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_same_tree(left, right)
    else:
        assert actual == expected, (actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-repo', type=Path, default=ROOT)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    source = subprocess.check_output(
        ['git', '-C', str(args.base_repo), 'show', f'{BASE}:{CORE}'],
        encoding='utf-8',
    )
    module = types.ModuleType('networks._no_a1a2_base')
    module.__package__ = 'networks'
    exec(compile(source, f'{BASE}/{CORE}', 'exec'), module.__dict__)

    torch.manual_seed(1234)
    with patch.object(vision_transformer, 'SwinTransformerSys', module.SwinTransformerSys):
        baseline = make_model(True).eval()
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(1234)
    ablation = make_model(True).eval()
    assert torch.equal(baseline_rng, torch.get_rng_state()), 'Initialization changed RNG'

    original, retained = baseline.state_dict(), ablation.state_dict()
    removed = set(original) - set(retained)
    assert removed and all(is_removed_key(key) for key in removed), sorted(removed)
    assert not (set(retained) - set(original)), 'Unexpected new model tensors'
    assert not any(is_removed_key(key) for key in retained)
    for key, tensor in retained.items():
        assert torch.equal(tensor, original[key]), f'Changed retained initialization: {key}'
    assert not any(type(m).__name__ == 'RoadAttentionHead' for m in ablation.modules())
    assert all(not hasattr(layer.downsample, 'road_alpha')
               for layer in ablation.swin_unet.layers if layer.downsample is not None)
    print('RETAINED_STATE_AND_RNG_EXACT_OK tensors=', len(retained), flush=True)
    print('PARAMETERS_REMOVED=', sum(p.numel() for p in baseline.parameters())
          - sum(p.numel() for p in ablation.parameters()), flush=True)

    # Disable only A1/A2 in the original implementation. The full H0 exchange,
    # encoder auxiliary tasks, decoder, global topology and Surface still run.
    for layer in baseline.swin_unet.layers:
        layer.road_attention_head = None
        for block in layer.blocks:
            block.use_road_bias = False
    baseline.to(args.device)
    ablation.to(args.device)
    image = torch.randn(1, 3, 512, 512, device=args.device)
    with torch.no_grad():
        expected = baseline(image)
        actual = ablation(image)
    assert actual[0].shape == (1, 1, 512, 512)
    assert torch.isfinite(actual[0]).all()
    assert len(actual[5]) == 3, 'Encoder road/skeleton tasks were removed'
    assert not any('road_attention' in item for item in actual[4])
    assert_same_tree(actual, expected)
    print('ALL_512_OUTPUTS_EXACT_VS_DISABLED_BASE_OK', flush=True)


if __name__ == '__main__':
    main()
