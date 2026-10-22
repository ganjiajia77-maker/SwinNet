"""Versioned, checkpointed settings for H0 local/gap connectivity."""
import json
from dataclasses import asdict, dataclass

EXPERIMENT_PROTOCOL = 'h0-local-gap-mass1500-fp32-noema80-v1'


@dataclass(frozen=True)
class LocalGapConfig:
    c_target: str = 'fine_edges'
    use_h0: bool = True
    p2_weight: float = 0.05
    gap_weight: float = 0.01
    threshold: float = 0.5
    min_distance: float = 3.0
    max_distance: float = 32.0
    tangent_cosine: float = 0.5
    trace_steps: int = 5
    min_trace: int = 3
    boundary_margin: int = 3
    min_component: int = 4
    max_endpoints: int = 128
    per_endpoint: int = 3
    max_edges: int = 64
    samples: int = 32
    side_offset: float = 1.0
    band_radius: float = 1.5
    gt_snap_radius: float = 8.0
    gt_corridor_radius: float = 8.0
    gt_detour_ratio: float = 1.5
    writeback_start: int = 10
    writeback_ramp: int = 5

    @classmethod
    def parse(cls, value=None):
        if isinstance(value, cls):
            return value
        data = json.loads(value) if isinstance(value, str) else (value or {})
        config = cls(**data)
        if config.c_target not in {'fine_edges', 'pooled_adjacency'}:
            raise ValueError('Unknown C target mode')
        if not isinstance(config.use_h0, bool):
            raise ValueError('use_h0 must be boolean')
        if not 0 < config.threshold < 1 or not -1 <= config.tangent_cosine <= 1:
            raise ValueError('Invalid candidate threshold/cosine')
        if not 0 < config.min_distance < config.max_distance:
            raise ValueError('Invalid candidate distance range')
        for name in ('trace_steps', 'min_trace', 'min_component', 'max_endpoints',
                     'per_endpoint', 'max_edges', 'samples', 'writeback_ramp'):
            value = getattr(config, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if config.samples < 2 or config.min_trace > config.trace_steps:
            raise ValueError('Invalid sampling/trace lengths')
        if config.boundary_margin < 1 or config.writeback_start < 0:
            raise ValueError('Invalid boundary margin/writeback start')
        for name in ('p2_weight', 'gap_weight', 'side_offset', 'band_radius',
                     'gt_snap_radius', 'gt_corridor_radius', 'gt_detour_ratio'):
            if getattr(config, name) <= 0:
                raise ValueError(f'{name} must be positive')
        return config

    def to_dict(self):
        return asdict(self)
