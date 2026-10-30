"""Pure segmentation + rebuilt Q + three-step source-selected repair.

No frozen BASE/Q3 dependency, independent RGB branch, S/C/D heads, global
topology, or active scalar G. Existing frozen-model code is left unchanged.
"""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from networks.q3_r3_repair import MultiScaleFeatures, make_q_target, dice_loss
from networks.reliable_directional_repair import (
    VisualProjection, DirectionRelations, MultiScaleGateRelation, DirectionalDynamics,
    directional_visual_sources, compose_surface_logits, dynamic_losses)
from networks.v9_source_selector import SourceSelectedRepair, SourceRelationSelector
from networks.v9_source_relation_labels import RelationLabels
from networks.v9_selector_transport_training import live_relation_loss
from networks.v9_late_choice_loss import late_choice_loss

VERSION = 'slim_end_to_end_source_transport_v1'


class SurfaceOnlyHead(nn.Module):
    """Exactly the four retained surface modules of the pure-segmentation control."""
    def __init__(self, original):
        super().__init__()
        for name in ('surface_proj', 'surface_branch', 'surface_refine', 'surface_head'):
            setattr(self, name, getattr(original, name))

    def forward(self, value):
        return self.surface_head(self.surface_refine(self.surface_branch(self.surface_proj(value))))


def prune_structure(core):
    """Same retained backbone/skip/CNN/DCA/surface modules as surface-only control."""
    for name in ('enable_highres_structure_stream','enable_e128_stage_fusion',
                 'enable_h3_surface_fusion','enable_global_topology',
                 'enable_post_refine_structure_interaction','use_stage3_global_context'):
        setattr(core, name, False)
    for name in ('encoder_stage1_road_attention_head','encoder_stage2_road_attention_head',
                 'prepatch_structure_encoder','highres_skeleton_adapter','highres_structure_skeleton_head',
                 'structure_surface_correction_head','stage2_topology_source','global_context_head',
                 'global_topology','h3_surface_proj','surface_h3_fusion','gap_connection'):
        setattr(core, name, None)
    core.decoder_structure_blocks = nn.ModuleDict()
    core.highres_structure_fusion = nn.ModuleDict()
    for layer in core.layers:
        layer.road_attention_head = None; layer.use_road_bias = False
        layer.road_attention_modulates_downsample = False
        if layer.downsample is not None: layer.downsample.register_parameter('road_alpha', None)
    for module in core.modules():
        for flag in ('use_road_bias','use_decoder_structure_bias'):
            if hasattr(module, flag): setattr(module, flag, False)
        for name in ('road_bias_scale_a1','road_bias_scale_a2','decoder_connectivity_bias_scale'):
            if name in module._parameters: module.register_parameter(name, None)
    core.guided_head = SurfaceOnlyHead(core.guided_head)


class SegmentationFeatures(nn.Module):
    def __init__(self, config):
        super().__init__()
        from networks.vision_transformer import SwinUnet
        self.segmenter = SwinUnet(config, num_classes=1, return_skeleton=True,
            structure_profile='stage23_boundary_0626', stage_skeleton_mode='direct',
            # Legacy constructor requires H0 to instantiate local-gap heads;
            # prune them immediately, before any optimizer or forward exists.
            enable_highres_structure_stream=True, enable_h3_surface_fusion=False,
            enable_global_topology=False, enable_gap_refinement=False,
            enable_stage2_local_propagation=False, remove_stage2_pre_topology_source=True)
        prune_structure(self.segmenter.swin_unet)
        print('[SLIM] retained pure segmentation CNN/Swin/skip/DCA/surface; removed S/C/D, A1/A2, H0 RGB stream, Gap and global topology',flush=True)
        self.detail_channels = config.MODEL.SWIN.EMBED_DIM
        self.semantic_channels = 2 * self.detail_channels

    @staticmethod
    def to_map(tokens, height, width):
        b, length, channels = tokens.shape
        if length != height*width: raise ValueError('Decoder tokens and feature grid differ')
        return tokens.transpose(1,2).reshape(b,channels,height,width)

    def forward(self, image):
        core = self.segmenter.swin_unet
        x, skips, attentions = core.forward_features(image)
        if attentions: raise RuntimeError('Pure segmentation unexpectedly predicted road attention')
        semantic = None; ph,pw = core.patches_resolution
        for index, layer in enumerate(core.layers_up):
            if index:
                skip = skips[3-index]
                height,width = ph//(2**(3-index)),pw//(2**(3-index))
                if index >= 2:
                    refined = core.dca_blocks[index-2](
                        deep=self.to_map(x,height,width), shallow=self.to_map(skip,height,width))
                    skip = refined.flatten(2).transpose(1,2)
                x = core.concat_back_dim[index](torch.cat((x,skip),-1))
            x = layer(x)
            if index == 1:
                semantic = self.to_map(x,ph//2,pw//2)
        detail = self.to_map(core.norm_up(x),ph,pw)
        # Retain the control's final PatchExpand and CNN surface head.
        full = self.to_map(core.up(detail.flatten(2).transpose(1,2)),4*ph,4*pw)
        z0 = core.guided_head(full)
        return {'z0':z0,'detail':detail,'semantic':semantic}


class SegmentationAndQ(nn.Module):
    def __init__(self, backbone):
        super().__init__(); self.backbone = backbone
        self.detail_project = nn.Conv2d(backbone.detail_channels,64,1)
        self.semantic_project = nn.Conv2d(backbone.semantic_channels,96,1)
        self.q_features = MultiScaleFeatures(64+96+1,32)
        self.q_head = nn.Conv2d(32,1,1)

    def forward(self, image):
        base = self.backbone(image)
        detail = self.detail_project(base['detail'])
        semantic = self.semantic_project(base['semantic'])
        size = detail.shape[-2:]
        context = F.interpolate(semantic,size=size,mode='bilinear',align_corners=False)
        p0 = F.interpolate(base['z0'].sigmoid(),size=size,mode='bilinear',align_corners=False)
        q_logits = self.q_head(self.q_features(torch.cat((detail,context,p0),1)))
        return dict(base,detail_projected=detail,semantic_projected=semantic,
                    q_logits=q_logits,q=q_logits.sigmoid())


class DirectionOnlyHead(nn.Module):
    """Eight trainable W rows; compatibility zero is not a gate parameter."""
    def __init__(self, old):
        super().__init__(); self.directions=nn.Conv2d(old.in_channels,8,1)
        with torch.no_grad():
            self.directions.weight.copy_(old.weight[1:]); self.directions.bias.copy_(old.bias[1:])

    def forward(self, value):
        directions = self.directions(value)
        return torch.cat((torch.zeros_like(directions[:,:1]),directions),1)


class ZeroDelta(nn.Module):
    def forward(self, value): return torch.zeros_like(value[:,:1])


class RepairCore(nn.Module):
    def __init__(self, upstream):
        super().__init__(); self.upstream = upstream
        self.visual = VisualProjection(160,32)
        self.reliable_head = nn.Sequential(nn.Conv2d(33,32,3,padding=1),nn.GELU(),nn.Conv2d(32,1,1))
        nn.init.zeros_(self.reliable_head[-1].weight); nn.init.constant_(self.reliable_head[-1].bias,1.5)
        self.relations = DirectionRelations(32,bridge_route=True,conditional_transport=True,transport_gate_mode='v7')
        self.relations.direction_head = DirectionOnlyHead(self.relations.direction_head)
        self.gate_relation = MultiScaleGateRelation(64,96,32)
        # Reuse projections and per-direction pair context, remove scalar-G weights.
        self.gate_relation.delta = ZeroDelta()
        self.proposal_head = nn.Sequential(nn.Conv2d(71,32,3,padding=1),nn.GELU(),nn.Conv2d(32,2,1))
        self.accept_head = nn.Sequential(nn.Conv2d(72,32,3,padding=1),nn.GELU(),nn.Conv2d(32,1,1))
        nn.init.zeros_(self.proposal_head[-1].weight)
        with torch.no_grad(): self.proposal_head[-1].bias.copy_(torch.tensor([-6.,-7.]))
        nn.init.zeros_(self.accept_head[-1].weight); nn.init.constant_(self.accept_head[-1].bias,-1.)
        self.dynamics = DirectionalDynamics(32,steps=3)


class SlimEndToEndRepair(SourceSelectedRepair):
    """Reuse the actual source-selected update, not the frozen parent forward."""
    def __init__(self, backbone):
        nn.Module.__init__(self)
        self.original = RepairCore(SegmentationAndQ(backbone))
        self.selector = SourceRelationSelector(32,32)
        self._context=None; self.mode='source'

    def train(self, mode=True):
        # Deliberately bypass SourceSelectedRepair.train's forced eval/freeze.
        return nn.Module.train(self,mode)

    def forward(self, image, capture_diagnostics=False):
        old = self.original.upstream(image)
        z0,q = old['z0'],old['q']
        highres = old['detail_projected']; size=highres.shape[-2:]
        decoder=F.interpolate(old['semantic_projected'],size=size,mode='bilinear',align_corners=False)
        p0=F.interpolate(z0.sigmoid(),size=size,mode='bilinear',align_corners=False)
        visual=self.original.visual(highres,decoder)
        reliable=p0*self.original.reliable_head(torch.cat((visual,p0),1)).sigmoid()
        relation=self.original.relations(visual,p0,reliable,return_evidence=True)
        weights,two,one=relation[:3]
        _,_,context=self.original.gate_relation(highres,decoder,visual,p0,q,reliable,weights,two,one,
                                               'full',capture_diagnostics=True)
        self._context={'feature':torch.cat((context['highres_projected'],context['decoder_projected']),1),
            'pair':context['per_direction_pair_raw'],'local':context['center'],'p0':p0,'q':q,'r':reliable,'w':weights}
        source_visual,coverage,_=directional_visual_sources(visual,reliable,weights,self.original.relations.scale_logits)
        bridge=(weights*two.repeat_interleave(2,1)).sum(1,keepdim=True)
        one_side=(weights*one.repeat_interleave(2,1)).sum(1,keepdim=True)
        support=(bridge+.25*one_side).clamp(0,1)
        positive_support=(support+.05*q*self.original.relations.semantic(visual).sigmoid()).clamp(0,1)
        features=torch.cat((visual,source_visual,p0,q,reliable,bridge,one_side,positive_support,coverage),1)
        raw=self.original.proposal_head(features)
        plus=8.*raw[:,:1].sigmoid(); minus=4.*raw[:,1:].sigmoid()
        proposal=(torch.logit(p0.clamp(1e-5,1.-1e-5))+positive_support*plus-minus).sigmoid()
        accept_logit=self.original.accept_head(torch.cat((features,proposal-p0),1))
        accept_logit=accept_logit+(bridge-one_side)/(bridge+one_side+.05)
        accept=accept_logit.sigmoid()*(1.-torch.sigmoid(10.*(proposal-p0))*(1.-(.1+.9*positive_support)))
        states,masses,available,logs=self._source_dynamics(visual,p0,q,reliable,proposal,accept,weights)
        z1,delta=compose_surface_logits(z0,p0,states[-1])
        result={'z0':z0,'z1':z1,'p0':z0.sigmoid(),'p1':z1.sigmoid(),'q':q,'q_logits':old['q_logits'],
            'r':reliable,'w':weights,'b':proposal,'a':accept,'u0':states[0],'u1':states[1],'u2':states[2],'u3':states[3],
            'states':states,'selector_context':self._context,'source_available':torch.stack(available,1),
            'source_mass':torch.stack(masses,1),'delta_logit':delta,'delta_z_plus':plus,'delta_z_minus':minus,**logs}
        if capture_diagnostics:
            result['feature_shapes']={key:list(old[key].shape) for key in ('detail','semantic','detail_projected','semantic_projected','q_logits')}
        return result


def p0_surface_loss(logits, target):
    bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    return (.75*(1.-torch.exp(-bce))*bce).mean()+.5*dice_loss(logits,target)


def end_to_end_losses(result, mask, skeleton, labels=RelationLabels(), seed=1234,
                      lambda_q=.1, lambda_choice=.05, lambda_p0=1.):
    base,terms=dynamic_losses(result,mask,skeleton,lambda_cont=.2,return_live_terms=True)
    relation,info=live_relation_loss(result,mask,skeleton,labels,seed)
    choice,counts=late_choice_loss(result,info)
    initial=p0_surface_loss(result['z0'],mask)
    # Only target generation is detached; Q input and predicted Q remain live.
    target=make_q_target(skeleton,result['z0'].detach(),result['q_logits'].shape[-2:])
    bce=F.binary_cross_entropy_with_logits(result['q_logits'],target,reduction='none')
    classes=[bce[take].mean() if bool(take.any()) else bce.sum()*0. for take in (target>.5,target<=.5)]
    q_loss=.5*sum(classes)+.5*dice_loss(result['q_logits'],target)
    total=base+lambda_p0*initial+lambda_q*q_loss+.1*relation+lambda_choice*choice
    terms.update(p0=initial.detach(),q_loss=q_loss.detach(),relation=relation.detach(),choice=choice.detach(),
        q_positive_fraction=target.mean().detach(),choice_counts=counts,relation_counts=info['matching'])
    return total,terms


def build_model(img_size=512, activation_checkpoint=False):
    from config import get_config
    from pathlib import Path
    config=get_config(SimpleNamespace(cfg=str(Path(__file__).resolve().parents[1]/'configs/swin_tiny_patch4_window7_224_lite.yaml'),
        opts=None,batch_size=2,img_size=img_size,zip=False,cache_mode=None,resume=None,accumulation_steps=1,
        use_checkpoint=activation_checkpoint,amp_opt_level=None,tag=None,eval=False,throughput=False))
    return SlimEndToEndRepair(SegmentationFeatures(config))
