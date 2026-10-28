"""Surface-only ablation with the original shared parameter initialization."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .keypoint_global_topology import KeypointGuidedGlobalTopology


class SurfaceGlobalTopology(KeypointGuidedGlobalTopology):
    """H0-style sparse global interaction using only surface features."""

    def __init__(self, channels, max_nodes=32, heads=4, alpha_max=0.05):
        super().__init__(channels, struct_channels=0, max_nodes=max_nodes,
                         heads=heads, alpha_max=alpha_max, enabled=True,
                         connectivity_channels=0)
        self.node_projection = nn.Linear(channels + 8 + 2, channels)
        self.register_parameter("connectivity_topology_bias_scale", None)
        self.neighbor_gate_lambda = 0.2
        self.neighbor_gate_kernel = 9

    def forward_surface(self, feature, surface_prob):
        batch, _, height, width = feature.shape
        # Selection and the probability gate are detached, as in H0.
        with torch.no_grad():
            p = surface_prob.detach().to(dtype=feature.dtype).clamp(0.0, 1.0)
            coords, valid, _, _ = self._extract_fps_anchors(p)
            near = F.max_pool2d(p, self.neighbor_gate_kernel, stride=1,
                                padding=self.neighbor_gate_kernel // 2)
            gate = p + self.neighbor_gate_lambda * (1.0 - p) * near
        sampled = self._sample_features_at_anchor_coords(feature, coords, (height, width))
        types = coords.new_zeros(batch, self.max_nodes)
        coords_norm = coords.float() / feature.new_tensor(
            [max(height - 1, 1), max(width - 1, 1)])
        nodes = self.node_projection(torch.cat(
            [sampled, self.node_type_embedding(types), coords_norm], dim=-1))
        nodes, _, _ = self._refine_tokens_with_relative_topology(
            nodes, coords, valid, (height, width), connectivity_feature=None)
        context = self._cross_attention_from_structure_tokens(feature, nodes, valid)
        return feature + self.alpha_global * gate * self.grid_projection(context)


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

    def _apply_post_refine_structure_interaction(self, feature, z_struct):
        return feature


def configure_surface_only(core, max_nodes=32, heads=4, alpha_max=0.05):
    # Prune after the original constructor consumes RNG, preserving shared tensors.
    core.surface_only = True
    for name in ("enable_highres_structure_stream", "enable_e128_stage_fusion",
                 "enable_h3_surface_fusion", "enable_global_topology",
                 "enable_post_refine_structure_interaction", "use_stage3_global_context"):
        setattr(core, name, False)
    for name in ("encoder_stage1_road_attention_head", "encoder_stage2_road_attention_head",
                 "prepatch_structure_encoder", "highres_skeleton_adapter", "highres_structure_skeleton_head",
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
    core.global_topology = SurfaceGlobalTopology(core.embed_dim, max_nodes, heads, alpha_max)
    core.enable_global_topology = True
    print("[ABLATION] H0 surface-only backbone + decoder-only global topology; "
          "neighbor gate lambda=0.2 kernel=9", flush=True)
