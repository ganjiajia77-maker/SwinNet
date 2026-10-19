"""Checkpoint identity for the original H0 Stage 2 location experiment."""

import hashlib
import json
from pathlib import Path


ARCHITECTURE_ID = "h0_stage2_location_v1"
BASE_MODEL_COMMIT = "599a410de8d341850e1810417307e4b3bbdfc82d"
ROOT = Path(__file__).resolve().parents[1]


def make_contract(config, constructor):
    required = dict(return_skeleton=True, structure_profile="stage23_boundary_0626",
                    enable_highres_structure_stream=True, highres_structure_channels=64,
                    highres_structure_fuse_stages="stage23", highres_structure_fusion_mode="stage23",
                    stage_skeleton_mode="direct", remove_stage2_pre_topology_source=True,
                    enable_global_topology=True, enable_h3_surface_fusion=True,
                    enable_post_refine_structure_interaction=False)
    wrong = {key: (constructor.get(key), value) for key, value in required.items()
             if constructor.get(key) != value}
    if wrong:
        raise ValueError(f"H0 location experiment requires original H0 configuration: {wrong}")
    swin = dict(config.MODEL.SWIN)
    if swin["EMBED_DIM"] != 96 or swin["PATCH_SIZE"] != 4:
        raise ValueError("H0 location experiment requires 96-channel, stride4 E1")
    protocol = ROOT / "mass_protocol.py"
    protocol_source = ROOT / "PROTOCOL_SOURCE_SHA256.json"
    if not protocol.is_file() or not protocol_source.is_file():
        raise FileNotFoundError("Missing versioned original H0 Massachusetts protocol")
    arguments = {key: value for key, value in constructor.items()
                 if key not in {"self", "config", "__class__"}}
    return json.loads(json.dumps(dict(
        architecture_id=ARCHITECTURE_ID,
        stage2_structure_location=constructor["stage2_structure_location"],
        base_model_commit=BASE_MODEL_COMMIT,
        protocol_version="h0-mass-native1500-random512-fp32-80e-bs2-v1",
        protocol_source_sha256=json.loads(protocol_source.read_text(encoding="utf-8")),
        protocol_implementation_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        model_config=dict(swin=swin, image_size=config.DATA.IMG_SIZE,
                          drop_rate=config.MODEL.DROP_RATE, drop_path_rate=config.MODEL.DROP_PATH_RATE,
                          use_checkpoint=config.TRAIN.USE_CHECKPOINT),
        constructor=arguments,
        H=dict(channels=64, stride=4, grid=[config.DATA.IMG_SIZE // 4] * 2),
        A1A2_enabled=True, global_topology_enabled=True,
        stage2_unit_executions=1, stage3_unit_executions=1,
    ), sort_keys=True))


def validate_checkpoint_contract(model, checkpoint):
    expected = getattr(model, "architecture_contract", None)
    actual = checkpoint.get("architecture_contract")
    if expected is None:
        if actual is not None:
            raise ValueError("Location-experiment checkpoint requires its explicit architecture entry")
        return
    if actual != expected:
        keys = set(expected) | set(actual or {})
        mismatch = [key for key in sorted(keys) if expected.get(key) != (actual or {}).get(key)]
        raise ValueError(f"H0 checkpoint architecture/location mismatch: {mismatch}; legacy H0 is not a resume")


def load_checkpoint_strict(model, checkpoint):
    validate_checkpoint_contract(model, checkpoint)
    return model.load_state_dict(checkpoint["model_state_dict"], strict=True)


def inherit_location_args(args, checkpoint, explicit):
    contract = checkpoint.get("architecture_contract")
    if contract is None and checkpoint.get("args", {}).get("architecture_id") == ARCHITECTURE_ID:
        raise ValueError("Location-experiment checkpoint is missing its architecture contract")
    for name in ("architecture_id", "stage2_structure_location"):
        saved = contract.get(name) if contract else None
        requested = getattr(args, name, None)
        if explicit(name) and requested != saved:
            raise ValueError(f"Requested {name}={requested} differs from checkpoint {saved}")
        if not explicit(name):
            setattr(args, name, saved)
    if getattr(args, "stage2_structure_location", None) == "encoder_e1" and not contract:
        raise ValueError("Original H0 checkpoint cannot resume encoder_e1")
