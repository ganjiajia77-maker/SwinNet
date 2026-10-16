"""Surface-only ablation with the original shared parameter initialization."""

import torch.nn as nn


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


def configure_surface_only(core):
    # Prune after the original constructor consumes RNG, preserving shared tensors.
    core.surface_only = True
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
