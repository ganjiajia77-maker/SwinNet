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

TOPOLOGY_ATTENTION_VERSION = "0621-gates-local-gap-topology-refiner-v4"


def get_topology_coefficients(model):
    module = model.module if hasattr(model, "module") else model
    swin_unet = module.swin_unet
    coefficients = {}

    for stage, structure_block in enumerate(
        swin_unet.decoder_structure_blocks
    ):
        coefficients[f"decoder_stage{stage}"] = {
            "gamma1": float(structure_block.gamma1.detach().cpu()),
            "gamma2": float(structure_block.gamma2.detach().cpu()),
        }
    topology_attention = swin_unet.guided_head.final_topology_attention
    coefficients["final_topology"] = {
        "raw_eta": float(topology_attention.raw_eta.detach().cpu()),
        "eta_eff": float(topology_attention.effective_eta().detach().cpu()),
        "raw_rho_gap": float(swin_unet.guided_head.raw_rho_gap.detach().cpu()),
        "rho_gap_eff": float(
            swin_unet.guided_head.effective_rho_gap().detach().cpu()
        ),
    }

    return coefficients


def format_topology_coefficients(model):
    coefficients = get_topology_coefficients(model)
    fields = [f"version={TOPOLOGY_ATTENTION_VERSION}"]
    for stage in (
        "decoder_stage0",
        "decoder_stage1",
        "decoder_stage2",
        "decoder_stage3",
    ):
        values = coefficients[stage]
        fields.append(
            "{} gamma1={:.6f} gamma2={:.6f}".format(
                stage,
                values["gamma1"],
                values["gamma2"],
            )
        )
    final_values = coefficients["final_topology"]
    fields.append(
        "final_topology raw_eta={:.6f} eta_eff={:.8f} "
        "raw_rho_gap={:.6f} rho_gap_eff={:.8f}".format(
            final_values["raw_eta"],
            final_values["eta_eff"],
            final_values["raw_rho_gap"],
            final_values["rho_gap_eff"],
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
    missing_final_topology = not any(
        "final_topology_attention." in key for key in state_dict
    )
    is_legacy_structure_checkpoint = (
        any("decoder_structure_blocks." in key for key in state_dict)
        and not any("raw_rho_gap" in key for key in state_dict)
    )
    if is_legacy_structure_checkpoint:
        result = model.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = (
            "swin_unet.guided_head.final_topology_attention.",
            "swin_unet.guided_head.raw_rho_gap",
            "swin_unet.guided_head.fixed_rho_gap",
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
        return result
    return model.load_state_dict(state_dict, strict=strict)


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
                 final_topology_eta_init=0.005, final_gap_rho_init=0.005):
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
                                final_gap_rho_init=final_gap_rho_init)
        
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

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        outputs = self.swin_unet(x)

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
 
