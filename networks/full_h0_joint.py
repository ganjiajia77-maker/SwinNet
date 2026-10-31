"""Original 599a410 H0, optionally with the validated live source transport.

The isolated reference package preserves the old architecture AND supervision.
No frozen-forward wrapper is used; GT is never an argument to model.forward.
"""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from networks.q3_r3_repair import MultiScaleFeatures
from networks.reliable_directional_repair import MultiScaleGateRelation
from networks.slim_end_to_end import SlimEndToEndRepair, RepairCore
from networks.v9_source_selector import SourceRelationSelector
from networks.v9_selector_transport_training import SplitDirectionHead, load_canonical

VERSION = 'full_h0_source_joint_v1'


def reference_config(size=512):
    from config import get_config
    return get_config(SimpleNamespace(
        cfg=str(Path(__file__).resolve().parents[1]/'configs/swin_tiny_patch4_window7_224_lite.yaml'),
        opts=None, batch_size=2, img_size=size, zip=False, cache_mode=None,
        resume=None, accumulation_steps=1, use_checkpoint=False, amp_opt_level=None,
        tag=None, eval=False, throughput=False))


def build_h0(saved, size=512):
    from reference_h0_599a410.networks.vision_transformer import SwinUnet
    defaults=json.loads((Path(__file__).resolve().parents[1]/'reference_h0_599a410/argument_defaults.json').read_text())
    args=dict(defaults, **saved)
    if not all(args.get(key) for key in ('enable_highres_structure_stream','enable_h3_surface_fusion','enable_global_topology')):
        raise ValueError('This experiment requires the complete H0: H stream, H3 fusion and global topology ON')
    kwargs={key:args[key] for key in inspect.signature(SwinUnet).parameters if key in args and key not in ('config','img_size','num_classes','return_skeleton')}
    return SwinUnet(reference_config(size),img_size=size,num_classes=1,return_skeleton=True,**kwargs),args


def original_criterion(saved, device):
    from reference_h0_599a410.training_contract import build_criterion,get_final_loss_weights
    args=SimpleNamespace(**saved)
    return build_criterion(args,get_final_loss_weights(args),device)


def h0_loss(criterion, outputs, batch):
    z,boundary,skeleton,connectivity,stages=outputs[:5]
    return criterion(z,surface_gt=batch['mask'],skeleton_gt=batch['skeleton'],
        skeleton_dilate_gt=batch['skeleton_dilate'],stage_outputs=stages,
        boundary_logits=boundary,skeleton_logits=skeleton,connectivity_logits=connectivity)


class FullH0Control(nn.Module):
    def __init__(self,base):
        super().__init__(); self.base=base

    def forward(self,image,capture_diagnostics=False):
        outputs=self.base(image); z=outputs[0]
        return {'z0':z,'z1':z,'p0':z.sigmoid(),'p1':z.sigmoid(),'h0_outputs':outputs}


class LiveH0Q(nn.Module):
    """Same Q3 inputs as the trained checkpoint, with live H0 and Q gradients."""
    def __init__(self,base,q_state):
        super().__init__(); self.base=base
        weight=q_state['q_features']['project.0.weight']
        self.q_features=MultiScaleFeatures(weight.shape[1],weight.shape[0])
        self.q_head=nn.Conv2d(weight.shape[0],1,1)
        self.q_features.load_state_dict(q_state['q_features'],strict=True)
        self.q_head.load_state_dict(q_state['q_head'],strict=True)
        self.last_outputs=None

    def forward(self,image):
        captured=[]
        handle=self.base.swin_unet.decoder_structure_blocks['3'].register_forward_hook(
            lambda module,inputs,output:captured.append(output[0]))
        try: outputs=self.base(image)
        finally: handle.remove()
        if len(captured)!=1: raise RuntimeError('Expected one original H0 Stage3 call')
        z0=outputs[0]; stages=outputs[4]
        stage2=next(item for item in stages if item.get('stage')==2 and item.get('skeleton') is not None)
        highres=self.base.swin_unet.last_stage_features['H2']; decoder=captured[0]
        def align(value):
            return F.interpolate(value,size=highres.shape[-2:],mode='bilinear',align_corners=False)
        features=torch.cat((align(decoder),highres,align(z0.sigmoid()),
            align(stage2['skeleton'].sigmoid()),align(stage2['connectivity'].sigmoid())),1)
        q_logits=self.q_head(self.q_features(features))
        self.last_outputs=outputs
        return {'z0':z0,'q':q_logits.sigmoid(),'q_logits':q_logits,
            'detail':highres,'semantic':decoder,'detail_projected':highres,'semantic_projected':decoder}


class FullH0Joint(SlimEndToEndRepair):
    """Reuse the exact source dynamics and head equations; retain COMPLETE H0."""
    def __init__(self,base,q_state,source,rate_bias=-8.):
        nn.Module.__init__(self)
        upstream=LiveH0Q(base,q_state)
        self.original=RepairCore(upstream)
        # Restore the trained nine-row compatibility head before splitting off
        # the unused scalar row. The eight W rows remain trainable.
        self.original.relations.direction_head=SplitDirectionHead(nn.Conv2d(40,9,1))
        self.original.gate_relation=MultiScaleGateRelation(64,96,32)
        expected={'visual','reliable_head','relations','proposal_head','accept_head','dynamics'}
        if set(source['modules'])!=expected: raise ValueError('Incomplete actual-transport repair weights')
        for name in expected: load_canonical(getattr(self.original,name),source['modules'][name])
        self.original.gate_relation.load_state_dict(source['gate_relation'],strict=True)
        self.selector=SourceRelationSelector(32,source['selector']['edge.0.weight'].shape[0])
        self.selector.load_state_dict(source['selector'],strict=True)
        self.original.gate_relation.delta.requires_grad_(False)
        self.original.relations.direction_head.scalar.requires_grad_(False)
        self.rate_bias=float(rate_bias)
        # Initialization only: reduce the existing transport/proposal rate
        # logits. No new gate, coefficient, loss, or update equation is added.
        with torch.no_grad(): self.original.dynamics.rates[-1].bias[1:3].add_(self.rate_bias)
        self._context=None; self.mode='source'

    @property
    def base(self): return self.original.upstream.base

    def forward(self,image,capture_diagnostics=False):
        result=super().forward(image,capture_diagnostics)
        result['h0_outputs']=self.original.upstream.last_outputs
        return result


def parameter_groups(model,h0_lr=5e-5,repair_lr=2e-4,repair_rates=None):
    backbone={id(p) for p in model.base.parameters() if p.requires_grad}
    groups=[]
    rates=repair_rates or {}
    pools={}
    for name,param in model.named_parameters():
        if not param.requires_grad: continue
        if id(param) in backbone: group,lr='H0',h0_lr
        elif name.startswith('original.upstream.'): group,lr='Q',h0_lr
        else:
            group=('selector' if name.startswith('selector.') else name.split('.')[1])
            group={'dynamics':'rates','gate_relation':'gate_feature_projections_and_pairs'}.get(group,group)
            lr=rates.get(group,repair_lr)
        pools.setdefault((group,lr),[]).append(param)
    for (name,lr),params in pools.items():
        groups.append({'name':name,'params':params,'lr':lr,'initial_lr':lr,'minimum_lr':lr*.1})
    actual=[id(p) for group in groups for p in group['params']]
    if len(actual)!=len(set(actual)) or set(actual)!={id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError('Optimizer coverage mismatch')
    return torch.optim.AdamW(groups,weight_decay=.0001)
