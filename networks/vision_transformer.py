# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from .dilated_asterisk import DilatedAsteriskWithDirections

logger = logging.getLogger(__name__)

TOPOLOGY_ATTENTION_VERSION = "gate-sc-graph-diffusion-v1"
STRUCTURE_PROFILE_FULL = "full"
STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626 = "stage23_boundary_0626"


def _core_swin_unet(model):
    module = model.module if hasattr(model, "module") else model
    return module.swin_unet


def set_soft_graph_eval_mode(
    model,
    use_soft_graph=True,
    lambda_scale=1.0,
    lambda_override=None,
    identity_dir_convs=False,
):
    """Eval-only overrides for final soft skeleton graph propagation."""
    swin_unet = _core_swin_unet(model)
    head = swin_unet.guided_head
    head.eval_use_soft_graph = bool(use_soft_graph)
    graph = getattr(head, "graph_propagation", None)
    if graph is not None:
        graph.set_eval_lambda(lambda_value=lambda_override, lambda_scale=lambda_scale)
        if identity_dir_convs:
            graph.reset_dir_convs_to_identity()
    lambda_eff = 0.0
    if graph is not None:
        if lambda_override is not None:
            lambda_eff = float(lambda_override)
        else:
            lambda_eff = float(
                (graph.effective_lambda() * graph.eval_lambda_scale).detach().cpu()
            )
    return {
        "use_soft_graph": bool(head.eval_use_soft_graph),
        "lambda_scale": float(lambda_scale),
        "lambda_override": lambda_override,
        "lambda_eff": lambda_eff,
        "identity_dir_convs": bool(identity_dir_convs),
    }


def configure_graph_diagnostics(model, enabled=True):
    swin_unet = _core_swin_unet(model)
    head = swin_unet.guided_head
    head.capture_graph_diagnostics = bool(enabled)
    if not enabled:
        head.last_graph_diagnostics = None
        graph = getattr(head, "graph_propagation", None)
        if graph is not None:
            graph.capture_diagnostics = False
            graph.last_diagnostics = None
            graph.last_export = None


def get_graph_propagation_state(model):
    swin_unet = _core_swin_unet(model)
    head = swin_unet.guided_head
    graph = getattr(head, "graph_propagation", None)
    if graph is None:
        return None
    return {
        "eval_use_soft_graph": bool(head.eval_use_soft_graph),
        "lambda_scale": float(graph.eval_lambda_scale),
        "lambda_eff": float(graph.effective_lambda().detach().cpu()),
        "raw_lambda": float(graph.raw_lambda.detach().cpu()),
    }


def get_topology_coefficients(model):
    module = model.module if hasattr(model, "module") else model
    swin_unet = module.swin_unet
    guided_head = swin_unet.guided_head
    coefficients = {
        "structure_profile": getattr(swin_unet, "structure_profile", "full"),
        "final_structure_enabled": bool(guided_head.enable_final_structure),
        "graph_prop_enabled": bool(getattr(guided_head, "enable_graph_prop", False)),
        "graph_diffusion_enabled": bool(
            getattr(swin_unet, "enable_graph_diffusion", False)
        ),
        "simple_c_diffusion_enabled": bool(
            getattr(swin_unet, "enable_simple_c_diffusion", False)
        ),
        "sc_graph_diffusion_enabled": bool(
            getattr(swin_unet, "enable_sc_graph_diffusion", False)
        ),
        "structure_gate_enabled": bool(getattr(swin_unet, "enable_structure_gate", True)),
        "decoder_attention_bias_enabled": bool(
            getattr(swin_unet, "enable_decoder_attention_bias", True)
        ),
        "stage_topology_stages": getattr(swin_unet, "stage_topology_stages", "none"),
        "stage_topology_active": {
            f"stage{stage}": bool(swin_unet._stage_topology_enabled(stage))
            for stage in range(swin_unet.num_layers)
        },
    }

    graph = getattr(guided_head, "graph_propagation", None)
    if graph is not None:
        coefficients["graph_propagation"] = {
            "lambda_eff": float(graph.effective_lambda().detach().cpu()),
            "lambda_scale": float(graph.eval_lambda_scale),
            "eval_use_soft_graph": bool(guided_head.eval_use_soft_graph),
        }

    for stage, structure_block in enumerate(swin_unet.decoder_structure_blocks):
        stage_values = {
            "gamma1": float(structure_block.gamma1.detach().cpu()),
            "gamma2": float(structure_block.gamma2.detach().cpu()),
            "structure_enabled": bool(swin_unet._decoder_structure_enabled(stage)),
            "structure_gate_enabled": bool(structure_block.enable_structure_gate),
            "graph_diffusion_enabled": bool(structure_block.enable_graph_diffusion),
            "simple_c_diffusion_enabled": bool(
                structure_block.enable_simple_c_diffusion
            ),
            "sc_graph_diffusion_enabled": bool(
                structure_block.enable_sc_graph_diffusion
            ),
        }
        graph_diffusion = getattr(structure_block, "graph_diffusion", None)
        simple_c_diffusion = getattr(structure_block, "simple_c_diffusion", None)
        sc_graph_diffusion = getattr(structure_block, "sc_graph_diffusion", None)
        if graph_diffusion is not None:
            stage_values["graph_gamma"] = float(
                graph_diffusion.gamma.detach().cpu()
            )
        if simple_c_diffusion is not None:
            stage_values["simple_c_gamma"] = float(
                simple_c_diffusion.gamma.detach().cpu()
            )
        if sc_graph_diffusion is not None:
            stage_values["sc_graph_gamma"] = float(
                sc_graph_diffusion.gamma.detach().cpu()
            )
        coefficients[f"decoder_stage{stage}"] = stage_values

    if guided_head.enable_final_structure:
        topology_attention = guided_head.final_topology_attention
        coefficients["final_topology"] = {
            "active": True,
            "raw_eta": float(topology_attention.raw_eta.detach().cpu()),
            "eta_eff": float(topology_attention.effective_eta().detach().cpu()),
            "raw_rho_gap": float(guided_head.raw_rho_gap.detach().cpu()),
            "rho_gap_eff": float(guided_head.effective_rho_gap().detach().cpu()),
        }
    else:
        coefficients["final_topology"] = {
            "active": False,
            "raw_eta": 0.0,
            "eta_eff": 0.0,
            "raw_rho_gap": 0.0,
            "rho_gap_eff": 0.0,
        }

    coefficients["stage_topology"] = {
        "enabled": swin_unet.stage_topology_stages != "none",
        "bias_mode": getattr(swin_unet, "stage_topology_bias_mode", "pairwise_skeleton"),
        "ratio": float(getattr(swin_unet, "stage_topology_ratio", 0.08)),
        "topo_clip": float(getattr(swin_unet, "stage_topology_topo_clip", 4.0)),
        **{
            f"stage{stage}_alpha_base": float(
                swin_unet.stage_topology_scales[str(stage)]
                .effective_topology_alpha()
                .detach()
                .cpu()
            )
            for stage in (2, 3)
        },
    }

    return coefficients


def apply_structure_profile_runtime(model):
    """Sync frozen topology flags after checkpoint load for 0626-style profiles."""
    module = model.module if hasattr(model, "module") else model
    swin_unet = module.swin_unet
    guided_head = swin_unet.guided_head
    if guided_head.enable_final_structure:
        return

    topology_attention = guided_head.final_topology_attention
    with torch.no_grad():
        topology_attention.fixed_eta.zero_()
        topology_attention.raw_eta.zero_()
        guided_head.fixed_rho_gap.zero_()
        guided_head.raw_rho_gap.zero_()
    topology_attention.raw_eta.requires_grad_(False)
    guided_head.raw_rho_gap.requires_grad_(False)


def freeze_backbone_train_graph_only(model):
    """Freeze encoder/decoder/heads; leave graph_propagation trainable."""
    module = model.module if hasattr(model, "module") else model
    for param in module.parameters():
        param.requires_grad = False

    graph = getattr(module.swin_unet.guided_head, "graph_propagation", None)
    if graph is None:
        raise RuntimeError(
            "freeze_backbone_train_graph_only requires enable_graph_prop=True"
        )
    for param in graph.parameters():
        param.requires_grad = True

    trainable = [name for name, p in module.named_parameters() if p.requires_grad]
    return trainable


def format_topology_coefficients(model):
    coefficients = get_topology_coefficients(model)
    fields = [
        f"version={TOPOLOGY_ATTENTION_VERSION}",
        f"profile={coefficients['structure_profile']}",
        f"final_structure={'on' if coefficients['final_structure_enabled'] else 'off'}",
        f"graph_prop={'on' if coefficients['graph_prop_enabled'] else 'off'}",
    ]
    if coefficients.get("graph_propagation"):
        graph_values = coefficients["graph_propagation"]
        fields.append(
            "graph_lambda_eff={:.6f}".format(graph_values["lambda_eff"])
        )
    for stage in (
        "decoder_stage0",
        "decoder_stage1",
        "decoder_stage2",
        "decoder_stage3",
    ):
        values = coefficients[stage]
        enabled = "active" if values.get("structure_enabled") else "bypass"
        fields.append(
            "{}[{}] gamma1={:.6f} gamma2={:.6f}".format(
                stage,
                enabled,
                values["gamma1"],
                values["gamma2"],
            )
        )
    final_values = coefficients["final_topology"]
    if final_values.get("active"):
        fields.append(
            "final_topology raw_eta={:.6f} eta_eff={:.8f} "
            "raw_rho_gap={:.6f} rho_gap_eff={:.8f}".format(
                final_values["raw_eta"],
                final_values["eta_eff"],
                final_values["raw_rho_gap"],
                final_values["rho_gap_eff"],
            )
        )
    else:
        fields.append("final_topology=disabled")
    stage_values = coefficients["stage_topology"]
    fields.append(
        "stage_topology stages={} enabled={} mode={} ratio={:.4f} "
        "topo_clip={:.2f} stage2_alpha_base={:.6f} stage3_alpha_base={:.6f}".format(
            coefficients["stage_topology_stages"],
            stage_values["enabled"],
            stage_values["bias_mode"],
            stage_values["ratio"],
            stage_values["topo_clip"],
            stage_values["stage2_alpha_base"],
            stage_values["stage3_alpha_base"],
        )
    )
    return " | ".join(fields)


def print_topology_coefficients(model, prefix="[TOPOLOGY]"):
    message = f"{prefix} {format_topology_coefficients(model)}"
    print(message, flush=True)
    return message


def load_topology_checkpoint_state(
    model,
    state_dict,
    checkpoint_version,
    strict=True,
):
    model_state = model.state_dict()
    obsolete_unexpected_suffixes = (
        "decoder_connectivity_value_scale",
    )
    filtered_state_dict = {}
    skipped_obsolete_keys = []
    skipped_shape_keys = []
    for key, value in state_dict.items():
        if key not in model_state and key.endswith(obsolete_unexpected_suffixes):
            skipped_obsolete_keys.append(key)
            continue
        if (
            key in model_state
            and value.shape != model_state[key].shape
            and key.endswith("structure_gate.0.weight")
        ):
            skipped_shape_keys.append(key)
            continue
        filtered_state_dict[key] = value
    if skipped_obsolete_keys:
        print(
            "[TOPOLOGY] Ignored obsolete checkpoint keys not used by this model: "
            + ", ".join(skipped_obsolete_keys),
            flush=True,
        )
    if skipped_shape_keys:
        print(
            "[TOPOLOGY] Reinitialized direction-gate input weights: "
            + ", ".join(skipped_shape_keys),
            flush=True,
        )
    if skipped_obsolete_keys or skipped_shape_keys:
        state_dict = filtered_state_dict

    module = model.module if hasattr(model, "module") else model
    swin_unet = module.swin_unet
    if getattr(swin_unet, "enable_graph_diffusion", False):
        result = model.load_state_dict(state_dict, strict=False)
        graph_diffusion_missing_prefixes = tuple(
            "swin_unet.decoder_structure_blocks.{}.graph_diffusion.".format(stage)
            for stage in range(len(swin_unet.decoder_structure_blocks))
        )
        allowed_missing_prefixes = graph_diffusion_missing_prefixes
        allowed_unexpected_suffixes = ()
        if not getattr(swin_unet, "enable_decoder_attention_bias", True):
            allowed_unexpected_suffixes = (
                "decoder_skeleton_bias_scale",
                "decoder_connectivity_bias_scale",
            )
        invalid_missing = [
            key
            for key in result.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        invalid_unexpected = [
            key
            for key in result.unexpected_keys
            if not any(key.endswith(suffix) for suffix in allowed_unexpected_suffixes)
        ]
        if invalid_missing or invalid_unexpected:
            raise RuntimeError(
                "Graph-diffusion checkpoint mismatch: "
                f"missing={invalid_missing}, unexpected={invalid_unexpected}"
            )
        if result.missing_keys:
            print(
                "[TOPOLOGY] Loaded checkpoint with graph diffusion additions; "
                "new parameters use runtime initialization: "
                + ", ".join(result.missing_keys),
                flush=True,
            )
        if result.unexpected_keys:
            print(
                "[TOPOLOGY] Ignored checkpoint keys unused by graph-diffusion runtime: "
                + ", ".join(result.unexpected_keys),
                flush=True,
            )
        apply_structure_profile_runtime(model)
        return result

    if getattr(swin_unet, "enable_simple_c_diffusion", False):
        result = model.load_state_dict(state_dict, strict=False)
        simple_c_missing_prefixes = tuple(
            "swin_unet.decoder_structure_blocks.{}.simple_c_diffusion.".format(stage)
            for stage in range(len(swin_unet.decoder_structure_blocks))
        )
        invalid_missing = [
            key
            for key in result.missing_keys
            if not key.startswith(simple_c_missing_prefixes)
        ]
        if invalid_missing or result.unexpected_keys:
            raise RuntimeError(
                "Simple-C-diffusion checkpoint mismatch: "
                f"missing={invalid_missing}, unexpected={result.unexpected_keys}"
            )
        if result.missing_keys:
            print(
                "[TOPOLOGY] Loaded checkpoint with simple C diffusion additions; "
                "new parameters use runtime initialization: "
                + ", ".join(result.missing_keys),
                flush=True,
            )
        apply_structure_profile_runtime(model)
        return result

    if getattr(swin_unet, "enable_sc_graph_diffusion", False):
        result = model.load_state_dict(state_dict, strict=False)
        sc_graph_missing_prefixes = tuple(
            "swin_unet.decoder_structure_blocks.{}.sc_graph_diffusion.".format(stage)
            for stage in range(len(swin_unet.decoder_structure_blocks))
        )
        invalid_missing = [
            key
            for key in result.missing_keys
            if not key.startswith(sc_graph_missing_prefixes)
        ]
        if invalid_missing or result.unexpected_keys:
            raise RuntimeError(
                "SC-graph-diffusion checkpoint mismatch: "
                f"missing={invalid_missing}, unexpected={result.unexpected_keys}"
            )
        if result.missing_keys:
            print(
                "[TOPOLOGY] Loaded checkpoint with SC graph diffusion additions; "
                "new parameters use runtime initialization: "
                + ", ".join(result.missing_keys),
                flush=True,
            )
        apply_structure_profile_runtime(model)
        return result

    missing_final_topology = not any(
        "final_topology_attention." in key for key in state_dict
    )
    is_legacy_structure_checkpoint = (
        any("decoder_structure_blocks." in key for key in state_dict)
        and not any("raw_rho_gap" in key for key in state_dict)
    )
    is_0626_checkpoint = any(
        "global_context_head." in key for key in state_dict
    )
    if is_legacy_structure_checkpoint:
        result = model.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = (
            "swin_unet.guided_head.final_topology_attention.",
            "swin_unet.guided_head.raw_rho_gap",
            "swin_unet.guided_head.fixed_rho_gap",
            "swin_unet.stage_topology_scales.",
            "swin_unet.stage2_topology_source.",
            "swin_unet.guided_head.graph_propagation.",
            "swin_unet.encoder_road_attention_head.",
            "swin_unet.encoder_stage1_road_attention_head.",
            "swin_unet.encoder_stage2_road_attention_head.",
            "swin_unet.encoder_road_attention_alpha",
            "swin_unet.layers.2.blocks.0.road_bias_scale_a1",
            "swin_unet.layers.2.blocks.0.road_bias_scale_a2",
            "swin_unet.layers.2.blocks.0.road_bias_scale",
            "swin_unet.layers.2.blocks.1.road_bias_scale_a1",
            "swin_unet.layers.2.blocks.1.road_bias_scale_a2",
            "swin_unet.layers.2.blocks.1.road_bias_scale",
            "swin_unet.layers_up.2.blocks.0.decoder_skeleton_bias_scale",
            "swin_unet.layers_up.2.blocks.0.decoder_connectivity_bias_scale",
            "swin_unet.layers_up.2.blocks.1.decoder_skeleton_bias_scale",
            "swin_unet.layers_up.2.blocks.1.decoder_connectivity_bias_scale",
            "swin_unet.layers_up.3.blocks.0.decoder_skeleton_bias_scale",
            "swin_unet.layers_up.3.blocks.0.decoder_connectivity_bias_scale",
            "swin_unet.layers_up.3.blocks.1.decoder_skeleton_bias_scale",
            "swin_unet.layers_up.3.blocks.1.decoder_connectivity_bias_scale",
            "swin_unet.bottleneck_context_fusion.scale_projections.",
            "swin_unet.bottleneck_context_fusion.scale_attention.",
            "swin_unet.decoder_structure_blocks.0.direction_head.",
            "swin_unet.decoder_structure_blocks.1.direction_head.",
            "swin_unet.decoder_structure_blocks.2.direction_head.",
            "swin_unet.decoder_structure_blocks.3.direction_head.",
            "swin_unet.decoder_structure_blocks.0.direction_gate.",
            "swin_unet.decoder_structure_blocks.1.direction_gate.",
            "swin_unet.decoder_structure_blocks.2.direction_gate.",
            "swin_unet.decoder_structure_blocks.3.direction_gate.",
            "swin_unet.decoder_structure_blocks.0.direction_gate_beta",
            "swin_unet.decoder_structure_blocks.1.direction_gate_beta",
            "swin_unet.decoder_structure_blocks.2.direction_gate_beta",
            "swin_unet.decoder_structure_blocks.3.direction_gate_beta",
            "swin_unet.decoder_structure_blocks.0.structure_gate.0.weight",
            "swin_unet.decoder_structure_blocks.1.structure_gate.0.weight",
            "swin_unet.decoder_structure_blocks.2.structure_gate.0.weight",
            "swin_unet.decoder_structure_blocks.3.structure_gate.0.weight",
            "swin_unet.decoder_structure_blocks.0.graph_diffusion.",
            "swin_unet.decoder_structure_blocks.1.graph_diffusion.",
            "swin_unet.decoder_structure_blocks.2.graph_diffusion.",
            "swin_unet.decoder_structure_blocks.3.graph_diffusion.",
            "swin_unet.decoder_structure_blocks.0.simple_c_diffusion.",
            "swin_unet.decoder_structure_blocks.1.simple_c_diffusion.",
            "swin_unet.decoder_structure_blocks.2.simple_c_diffusion.",
            "swin_unet.decoder_structure_blocks.3.simple_c_diffusion.",
            "swin_unet.decoder_structure_blocks.0.sc_graph_diffusion.",
            "swin_unet.decoder_structure_blocks.1.sc_graph_diffusion.",
            "swin_unet.decoder_structure_blocks.2.sc_graph_diffusion.",
            "swin_unet.decoder_structure_blocks.3.sc_graph_diffusion.",
            "swin_unet.stage2_topology_source.direction_head.",
            "swin_unet.stage2_topology_source.direction_gate.",
            "swin_unet.stage2_topology_source.direction_gate_beta",
            "swin_unet.stage2_topology_source.structure_gate.0.weight",
        )
        if is_0626_checkpoint:
            allowed_missing_prefixes = allowed_missing_prefixes + (
                "swin_unet.guided_head.alpha",
                "swin_unet.guided_head.structure_branch.",
                "swin_unet.guided_head.skeleton_head.",
                "swin_unet.guided_head.connectivity_head.",
                "swin_unet.guided_head.structure_fusion.",
                "swin_unet.guided_head.structure_residual.",
            )
        invalid_missing = [
            key
            for key in result.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if invalid_missing or result.unexpected_keys:
            raise RuntimeError(
                "Legacy structure checkpoint mismatch: "
                f"missing={invalid_missing}, unexpected={result.unexpected_keys}"
            )
        module = model.module if hasattr(model, "module") else model
        guided_head = module.swin_unet.guided_head
        with torch.no_grad():
            guided_head.fixed_rho_gap.zero_()
            guided_head.raw_rho_gap.zero_()
        guided_head.raw_rho_gap.requires_grad_(False)
        if missing_final_topology:
            topology_attention = guided_head.final_topology_attention
            with torch.no_grad():
                topology_attention.fixed_eta.zero_()
                topology_attention.raw_eta.zero_()
            topology_attention.raw_eta.requires_grad_(False)
        print(
            "[TOPOLOGY] Loaded legacy structure checkpoint; missing final "
            "topology parameters use runtime initialization and rho_gap is "
            "disabled for output compatibility.",
            flush=True,
        )
        if is_0626_checkpoint:
            print(
                "[TOPOLOGY] Detected 0626 checkpoint with global_context_head; "
                "unused final-structure guided_head modules keep init values.",
                flush=True,
            )
        apply_structure_profile_runtime(model)
        return result
    road_attention_missing_prefixes = (
        "swin_unet.encoder_road_attention_head.",
        "swin_unet.encoder_stage1_road_attention_head.",
        "swin_unet.encoder_stage2_road_attention_head.",
        "swin_unet.encoder_road_attention_alpha",
        "swin_unet.layers.2.blocks.0.road_bias_scale_a1",
        "swin_unet.layers.2.blocks.0.road_bias_scale_a2",
        "swin_unet.layers.2.blocks.0.road_bias_scale",
        "swin_unet.layers.2.blocks.1.road_bias_scale_a1",
        "swin_unet.layers.2.blocks.1.road_bias_scale_a2",
        "swin_unet.layers.2.blocks.1.road_bias_scale",
        "swin_unet.layers_up.2.blocks.0.decoder_skeleton_bias_scale",
        "swin_unet.layers_up.2.blocks.0.decoder_connectivity_bias_scale",
        "swin_unet.layers_up.2.blocks.1.decoder_skeleton_bias_scale",
        "swin_unet.layers_up.2.blocks.1.decoder_connectivity_bias_scale",
        "swin_unet.layers_up.3.blocks.0.decoder_skeleton_bias_scale",
        "swin_unet.layers_up.3.blocks.0.decoder_connectivity_bias_scale",
        "swin_unet.layers_up.3.blocks.1.decoder_skeleton_bias_scale",
        "swin_unet.layers_up.3.blocks.1.decoder_connectivity_bias_scale",
        "swin_unet.decoder_structure_blocks.0.direction_head.",
        "swin_unet.decoder_structure_blocks.1.direction_head.",
        "swin_unet.decoder_structure_blocks.2.direction_head.",
        "swin_unet.decoder_structure_blocks.3.direction_head.",
        "swin_unet.decoder_structure_blocks.0.direction_gate.",
        "swin_unet.decoder_structure_blocks.1.direction_gate.",
        "swin_unet.decoder_structure_blocks.2.direction_gate.",
        "swin_unet.decoder_structure_blocks.3.direction_gate.",
        "swin_unet.decoder_structure_blocks.0.direction_gate_beta",
        "swin_unet.decoder_structure_blocks.1.direction_gate_beta",
        "swin_unet.decoder_structure_blocks.2.direction_gate_beta",
        "swin_unet.decoder_structure_blocks.3.direction_gate_beta",
        "swin_unet.decoder_structure_blocks.0.structure_gate.0.weight",
        "swin_unet.decoder_structure_blocks.1.structure_gate.0.weight",
        "swin_unet.decoder_structure_blocks.2.structure_gate.0.weight",
        "swin_unet.decoder_structure_blocks.3.structure_gate.0.weight",
        "swin_unet.decoder_structure_blocks.0.graph_diffusion.",
        "swin_unet.decoder_structure_blocks.1.graph_diffusion.",
        "swin_unet.decoder_structure_blocks.2.graph_diffusion.",
        "swin_unet.decoder_structure_blocks.3.graph_diffusion.",
        "swin_unet.decoder_structure_blocks.0.simple_c_diffusion.",
        "swin_unet.decoder_structure_blocks.1.simple_c_diffusion.",
        "swin_unet.decoder_structure_blocks.2.simple_c_diffusion.",
        "swin_unet.decoder_structure_blocks.3.simple_c_diffusion.",
        "swin_unet.decoder_structure_blocks.0.sc_graph_diffusion.",
        "swin_unet.decoder_structure_blocks.1.sc_graph_diffusion.",
        "swin_unet.decoder_structure_blocks.2.sc_graph_diffusion.",
        "swin_unet.decoder_structure_blocks.3.sc_graph_diffusion.",
        "swin_unet.stage2_topology_source.direction_head.",
        "swin_unet.stage2_topology_source.direction_gate.",
        "swin_unet.stage2_topology_source.direction_gate_beta",
        "swin_unet.stage2_topology_source.structure_gate.0.weight",
    )
    msaf_missing_prefixes = (
        "swin_unet.bottleneck_context_fusion.scale_projections.",
        "swin_unet.bottleneck_context_fusion.scale_attention.",
    )
    allowed_new_missing_prefixes = (
        road_attention_missing_prefixes + msaf_missing_prefixes
    )
    if strict and not any(
        key.startswith(allowed_new_missing_prefixes)
        for key in state_dict
    ):
        result = model.load_state_dict(state_dict, strict=False)
        invalid_missing = [
            key
            for key in result.missing_keys
            if not key.startswith(allowed_new_missing_prefixes)
        ]
        if invalid_missing or result.unexpected_keys:
            raise RuntimeError(
                "Checkpoint mismatch after allowing new encoder additions: "
                f"missing={invalid_missing}, unexpected={result.unexpected_keys}"
            )
        print(
            "[TOPOLOGY] Loaded checkpoint without new encoder additions; "
            "new parameters use runtime initialization.",
            flush=True,
        )
    else:
        result = model.load_state_dict(state_dict, strict=strict)
    apply_structure_profile_runtime(model)
    return result


def load_state_dict_ignore_mismatch(model, state_dict, prefix=""):
    model_dict = model.state_dict()
    filtered_dict = {}
    skipped_keys = []

    for key, value in state_dict.items():
        if key in model_dict and model_dict[key].shape == value.shape:
            filtered_dict[key] = value
        else:
            skipped_keys.append(key)

    model_dict.update(filtered_dict)
    msg = model.load_state_dict(model_dict, strict=False)
    print(f"{prefix}loaded keys: {len(filtered_dict)}")
    print(f"{prefix}skipped keys: {len(skipped_keys)}")
    for key in skipped_keys[:20]:
        print(f"{prefix}skipped: {key}")
    return msg


class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False,
                 use_asterisk=False, return_skeleton=False, bottleneck_type="global_local",
                 final_topology_eta_init=0.005, final_gap_rho_init=0.005,
                 stage_topology_stages="none",
                 stage_topology_alpha_max=1.0,
                 stage_topology_alpha_init=0.1,
                 stage_topology_bias_mode="pairwise_skeleton",
                 stage_topology_ratio=0.08,
                 stage_topology_topo_clip=4.0,
                 structure_profile=STRUCTURE_PROFILE_FULL,
                 enable_final_graph_prop=False,
                 enable_graph_diffusion=False,
                 enable_simple_c_diffusion=False,
                 enable_sc_graph_diffusion=False,
                 enable_structure_gate=True,
                 enable_decoder_attention_bias=True):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.use_asterisk = use_asterisk
        self.return_skeleton = return_skeleton
        self.bottleneck_type = bottleneck_type

        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                return_skeleton=self.return_skeleton,
                                bottleneck_type=self.bottleneck_type,
                                final_topology_eta_init=final_topology_eta_init,
                                final_gap_rho_init=final_gap_rho_init,
                                stage_topology_stages=stage_topology_stages,
                                stage_topology_alpha_max=stage_topology_alpha_max,
                                stage_topology_alpha_init=stage_topology_alpha_init,
                                stage_topology_bias_mode=stage_topology_bias_mode,
                                stage_topology_ratio=stage_topology_ratio,
                                stage_topology_topo_clip=stage_topology_topo_clip,
                                structure_profile=structure_profile,
                                enable_final_graph_prop=enable_final_graph_prop,
                                enable_graph_diffusion=enable_graph_diffusion,
                                enable_simple_c_diffusion=enable_simple_c_diffusion,
                                enable_sc_graph_diffusion=enable_sc_graph_diffusion,
                                enable_structure_gate=enable_structure_gate,
                                enable_decoder_attention_bias=enable_decoder_attention_bias)
        
        # Keep DilatedAsterisk in the graph, but turn off its residual effect.
        if self.use_asterisk:
            self.asterisk = DilatedAsteriskWithDirections(
                in_channels=self.num_classes,
                out_channels=self.num_classes,
                alpha_init=0.0,
            )
            self._disable_asterisk_alpha()
            print("[INFO] DilatedAsterisk code kept but disabled: alpha=0 (in_channels={})".format(self.num_classes))

    def _disable_asterisk_alpha(self):
        if self.use_asterisk and hasattr(self, "asterisk") and hasattr(self.asterisk, "alpha"):
            with torch.no_grad():
                self.asterisk.alpha.zero_()
            self.asterisk.alpha.requires_grad_(False)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._disable_asterisk_alpha()
        return result

    def forward(
        self,
        x,
        gt_skeleton=None,
        topology_alpha_scale=1.0,
        teacher_forcing_ratio=0.0,
    ):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        outputs = self.swin_unet(
            x,
            gt_skeleton=gt_skeleton,
            topology_alpha_scale=topology_alpha_scale,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        if isinstance(outputs, tuple):
            logits = outputs[0]
            aux_outputs = outputs[1:]
        else:
            logits = outputs
            aux_outputs = ()

        if self.use_asterisk:
            logits = self.asterisk(logits)

        if self.return_skeleton:
            return (logits, *aux_outputs)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = load_state_dict_ignore_mismatch(self.swin_unet, pretrained_dict)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if ".reduction.weight" in k and ".linear_reduction.weight" not in k:
                    linear_key = k.replace(".reduction.weight", ".linear_reduction.weight")
                    if linear_key not in full_dict:
                        full_dict[linear_key] = full_dict[k]
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, full_dict[k].shape, model_dict[k].shape))
                        del full_dict[k]

            msg = load_state_dict_ignore_mismatch(self.swin_unet, full_dict)
            # print(msg)
        else:
            print("none pretrain")
 
