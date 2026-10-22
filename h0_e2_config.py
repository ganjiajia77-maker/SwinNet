"""Versioned E2-only experiment configuration and strict checkpoint contract."""
import sys

from h0_encoder_protocol import BASE_COMMIT, DIRECT_BASE_COMMIT, MODEL_NAME, validate_training_args
from losses.encoder_direction_loss import DEFAULT_DIRECTION_CONFIG, validate_direction_config

AGGREGATION_CONFIG = dict(stage='E2', channels=192, radii=[1, 2],
                          initial_gamma=.05, initial_q=.5,
                          symmetric=True, align_corners=False,
                          boundary='both endpoints valid; renormalize radii; mask projected delta')


def add_direction_arguments(parser):
    for key, value in DEFAULT_DIRECTION_CONFIG.items():
        parser.add_argument('--' + key, type=type(value), default=value)


def direction_config(args):
    return validate_direction_config({key: getattr(args, key) for key in DEFAULT_DIRECTION_CONFIG})


def validate_checkpoint(checkpoint):
    if checkpoint.get('architecture_id') != MODEL_NAME:
        raise ValueError('Expected new E2/no-decoder-D architecture; old checkpoints require a fresh experiment')
    if checkpoint.get('base_commit') != BASE_COMMIT or checkpoint.get('direct_base_commit') != DIRECT_BASE_COMMIT:
        raise ValueError('Wrong H0 experiment baseline')
    if checkpoint.get('ema_state_dict') is not None:
        raise ValueError('Expected FP32 raw weights, no EMA')
    if checkpoint.get('encoder_aggregation_config') != AGGREGATION_CONFIG:
        raise ValueError('Unexpected E2 aggregation configuration')
    saved = checkpoint['args']
    validate_training_args(saved)
    cfg = validate_direction_config(checkpoint['encoder_direction_config'])
    if any(saved.get(key) != value for key, value in cfg.items()):
        raise ValueError('Direction config differs from saved arguments')
    return saved


def inherit_checkpoint_args(args, checkpoint, training=False):
    saved = validate_checkpoint(checkpoint)
    # Inherit all model/optimizer/loss settings before building either model or loss.
    # Runtime locations and evaluation output controls remain caller-owned.
    locations = {'root_path', 'output_dir', 'run_name', 'resume', 'model_path',
                 'pretrain_ckpt', 'num_workers', 'cfg'}
    eval_runtime = {'split', 'threshold', 'export_thresholds', 'overlap_infer',
                    'save_encoder_diagnostics', 'batch_size'}
    for name in vars(args).copy():
        if name not in saved or name in locations or (not training and name in eval_runtime):
            continue
        explicit = any(arg == '--' + name or arg.startswith('--' + name + '=')
                       for arg in sys.argv[1:])
        if explicit and getattr(args, name) != saved[name]:
            raise ValueError(f'Checkpoint conflict: --{name}={getattr(args, name)!r}, saved={saved[name]!r}')
        setattr(args, name, saved[name])
    return args
