"""Identity and strict checkpoint contract for the H0 skeleton feature experiment."""

ARCHITECTURE_ID = "h0_skeleton_feature_stream_v1"
BASE_COMMIT = "599a410de8d341850e1810417307e4b3bbdfc82d"

SAVED_CONFIG_KEYS = (
    "img_size", "source_patch_size", "bottleneck_type", "structure_profile",
    "enable_highres_structure_stream", "highres_structure_channels",
    "highres_structure_fuse_stages", "highres_structure_fusion_mode",
    "enable_h3_surface_fusion", "enable_post_refine_structure_interaction",
    "enable_global_topology", "stage_skeleton_mode", "remove_stage2_pre_topology_source",
    "stage2_skeleton_gradient_ratio", "stage3_skeleton_gradient_ratio",
    "stage3_gate_topology_gradient_ratio", "final_skeleton_gradient_ratio", "disable_msfe_skip",
    "stage2_skeleton_weight", "stage3_skeleton_weight", "stage_skeleton_loss_factor",
    "stage_skeleton_only_loss_factor", "highres_structure_skeleton_weight",
    "stage_direction_factor", "stage_connectivity_factor", "skeleton_pos_weight",
)


def restore_saved_config(args, checkpoint, cli_has):
    require_checkpoint_architecture(checkpoint)
    saved = checkpoint["args"]
    for key in SAVED_CONFIG_KEYS:
        if not hasattr(args, key) or key not in saved:
            continue
        if cli_has(key) and getattr(args, key) != saved[key]:
            raise ValueError(f"Checkpoint config conflict for {key}: CLI={getattr(args, key)}, saved={saved[key]}")
        setattr(args, key, saved[key])


def restore_model_config(config, saved_config):
    """Restore the full Swin configuration, including settings without state tensors."""
    if not isinstance(saved_config, str):
        raise ValueError("Checkpoint must contain its complete model_config YAML")
    from yacs.config import CfgNode
    config.defrost()
    config.merge_from_other_cfg(CfgNode.load_cfg(saved_config))
    config.freeze()
    return config


def architecture_metadata(model, args):
    core = model.swin_unet
    return {
        "id": ARCHITECTURE_ID,
        "base_commit": BASE_COMMIT,
        "enabled": True,
        "feature_channels": core.highres_structure_channels,
        "feature_hw": list(core.patches_resolution),
        "interaction": "synchronous_BH_HB_residual_then_delta_B_to_G",
        "surface_fusion": "pre_x4_concat_decoder96_G3proj32_FH2proj32_zero_delta",
        "global_topology": False,
        "skeleton_effective_weights": {
            "S2": args.stage2_skeleton_weight * args.stage_skeleton_loss_factor
            * args.stage_skeleton_only_loss_factor,
            "S3": args.stage3_skeleton_weight * args.stage_skeleton_loss_factor
            * args.stage_skeleton_only_loss_factor,
            "SH": args.highres_structure_skeleton_weight,
        },
        "interaction_strengths": {
            str(stage): {
                "alpha": float(core.decoder_structure_blocks[str(stage)]
                               .skeleton_feature_interaction.alpha.detach().cpu()),
                "beta": float(core.decoder_structure_blocks[str(stage)]
                              .skeleton_feature_interaction.beta.detach().cpu()),
            } for stage in (2, 3)
        },
    }


def require_checkpoint_architecture(checkpoint):
    meta = checkpoint.get("skeleton_feature_architecture", {}) if isinstance(checkpoint, dict) else {}
    if meta.get("id") != ARCHITECTURE_ID or meta.get("global_topology") is not False:
        raise ValueError(
            f"Expected {ARCHITECTURE_ID} checkpoint with global topology removed; "
            "old H0 checkpoints cannot be resumed or evaluated as this architecture."
        )
    saved = checkpoint.get("args", {})
    if (saved.get("enable_global_topology") or not saved.get("enable_highres_structure_stream")
            or saved.get("stage_skeleton_mode") != "direct"):
        raise ValueError("Checkpoint architecture flags conflict with skeleton feature stream")
    if not isinstance(checkpoint.get("model_config"), str):
        raise ValueError("Checkpoint has no complete saved model configuration")
    return meta
