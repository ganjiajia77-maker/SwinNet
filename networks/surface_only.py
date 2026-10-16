"""Surface-only ablation with the original shared parameter initialization."""

import torch.nn as nn

from .skeleton_guided_head import scale_gradient


SKELETON_AUX_ARCHITECTURE = "e202_surface_skeleton_aux_v1"


def skeleton_aux_metadata(core, args):
    if not getattr(core, "skeleton_aux_only", False):
        return None
    return dict(
        architecture=SKELETON_AUX_ARCHITECTURE,
        surface_baseline_commit="eb2da24d768b7f643e4fb288e3736693862c1fe8",
        stages=[2, 3], feature_hw=[args.img_size // 4] * 2, feature_channels=96,
        inputs="layer_up output, before any structure processing",
        head="original independent structure_branch + skeleton_head",
        feedback=False, stage_transfer=False, surface_fusion=False,
        gradient_ratios={str(i): core.skeleton_aux_heads[str(i)].gradient_ratio for i in (2, 3)},
        weights={"S2": args.stage2_skeleton_weight, "S3": args.stage3_skeleton_weight},
        target="original Gaussian soft stage skeleton target, BCE+Dice, pos_weight2",
    )


def require_skeleton_aux_checkpoint(checkpoint):
    saved = checkpoint.get("args", {})
    metadata = checkpoint.get("skeleton_aux_protocol") or {}
    if (not saved.get("surface_only") or not saved.get("skeleton_aux_only")
            or metadata.get("architecture") != SKELETON_AUX_ARCHITECTURE):
        raise ValueError("Expected the independent S2/S3 auxiliary experiment checkpoint; "
                         "do not resume from the trained surface-only baseline")
    return metadata


class SkeletonAuxiliaryHead(nn.Module):
    """The original structure/skeleton submodules, with no feedback outputs."""
    def __init__(self, original):
        super().__init__()
        self.structure_branch = original.structure_branch
        self.skeleton_head = original.skeleton_head
        self.gradient_ratio = original.skeleton_gradient_ratio

    def forward(self, feature):
        feature = scale_gradient(feature, self.gradient_ratio)
        return self.skeleton_head(self.structure_branch(feature))


class SurfaceOnlyHead(nn.Module):
    def __init__(self, original):
        super().__init__()
        for name in ("surface_proj", "surface_branch", "surface_refine", "surface_head"):
            setattr(self, name, getattr(original, name))
        self.enable_final_structure = False
        self.enable_final_skeleton_aux = False
        self.enable_post_refine_structure_interaction = False
        self.last_surface_pre_logits = None
        self.last_delta_logit = None
        self.last_graph_delta = None

    def forward(self, x, z_struct=None):
        feature = self.surface_refine(self.surface_branch(self.surface_proj(x)))
        logits = self.surface_head(feature)
        self.last_surface_pre_logits = logits
        return logits, None, None, None


def configure_surface_only(core, skeleton_aux_only=False):
    # Prune after the original constructor consumes RNG, preserving shared tensors.
    # Reuse the already initialized heads before their parent blocks are removed.
    # The wrappers allocate no random tensors and preserve the baseline RNG state.
    if skeleton_aux_only:
        core.skeleton_aux_heads = nn.ModuleDict({
            str(stage): SkeletonAuxiliaryHead(core.decoder_structure_blocks[str(stage)])
            for stage in (2, 3)
        })
    core.surface_only = True
    core.skeleton_aux_only = bool(skeleton_aux_only)
    for name in ("enable_highres_structure_stream", "enable_e128_stage_fusion",
                 "enable_h3_surface_fusion", "enable_global_topology",
                 "enable_post_refine_structure_interaction", "use_stage3_global_context"):
        setattr(core, name, False)
    for name in ("encoder_stage1_road_attention_head", "encoder_stage2_road_attention_head",
                 "prepatch_structure_encoder", "highres_structure_skeleton_head",
                 "structure_surface_correction_head", "stage2_topology_source",
                 "global_context_head", "global_topology", "h3_surface_proj", "surface_h3_fusion"):
        setattr(core, name, None)
    core.decoder_structure_blocks = nn.ModuleDict()
    core.highres_structure_fusion = nn.ModuleDict()
    for layer in core.layers:
        layer.road_attention_head = None
        layer.use_road_bias = False
        layer.road_attention_modulates_downsample = False
        if layer.downsample is not None:
            layer.downsample.register_parameter("road_alpha", None)
    for module in core.modules():
        if hasattr(module, "use_road_bias"):
            module.use_road_bias = False
        if hasattr(module, "use_decoder_structure_bias"):
            module.use_decoder_structure_bias = False
        for name in ("road_bias_scale_a1", "road_bias_scale_a2", "decoder_connectivity_bias_scale"):
            if name in module._parameters:
                module.register_parameter(name, None)
    core.guided_head = SurfaceOnlyHead(core.guided_head)
    print("[ABLATION] surface_only: A1/A2, S/C/D, structure feedback, high-res/E128, "
          "H3 fusion, boundary residual and global topology removed", flush=True)
    if skeleton_aux_only:
        print("[ABLATION] independent S2/S3 auxiliary heads after layer_up; "
              "no feedback/transfer; shared decoder gradient ratios=0.5/0.5", flush=True)
