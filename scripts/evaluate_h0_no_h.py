"""Same validation/test/six-metric protocol; strict complete-no-H checkpoint only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_h0_encoder_phase as evaluation

_base_build = evaluation.build_model


def build_model(saved):
    if not saved.get('complete_no_h_ablation'):
        raise ValueError('Specify a checkpoint from the complete-no-H ablation')
    return _base_build(saved)


if __name__ == '__main__':
    evaluation.NAME = 'h0_complete_no_h'
    evaluation.build_model = build_model
    evaluation.main()
