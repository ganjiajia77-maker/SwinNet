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
                 use_asterisk=False, return_skeleton=False, bottleneck_type="global_local"):
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
                                bottleneck_type=self.bottleneck_type)
        
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
 
