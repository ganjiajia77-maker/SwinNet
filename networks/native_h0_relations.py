"""Stage-2 global context and one Stage-3 local feature aggregation."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from topology_direction_constants import CONNECTIVITY_DIRECTIONS, CONNECTIVITY_OPPOSITE
from .skeleton_guided_head import SkeletonSpatialHead, scale_gradient


def shifted(x,dy,dx):
    height,width=x.shape[-2:]
    padded=F.pad(x,(max(-dx,0),max(dx,0),max(-dy,0),max(dy,0)))
    return padded[...,max(dy,0):max(dy,0)+height,max(dx,0):max(dx,0)+width]


class NativeStage3(nn.Module):
    def __init__(self, original, highres_channels=64, semantic_channels=96):
        super().__init__()
        self.structure_branch=original.structure_branch
        self.previous_structure_fusion=original.previous_structure_fusion
        self.highres_skeleton_fusion=original.highres_skeleton_fusion
        self.connectivity_context=original.connectivity_context
        self.connectivity_head=original.connectivity_head
        self.direction_head=original.direction_head
        self.skeleton_gradient_ratio=original.skeleton_gradient_ratio
        self.skeleton_head=SkeletonSpatialHead(highres_channels)
        self.h_project=nn.Conv2d(highres_channels,16,1)
        self.g_project=nn.Conv2d(semantic_channels,16,1)
        self.edge=nn.Sequential(nn.Conv2d(104,32,1),nn.ReLU(),nn.Conv2d(32,1,1))
        self.stay=nn.Conv2d(32,1,1)
        self.out=nn.Conv2d(highres_channels,highres_channels,1,bias=False)
        self.gamma=nn.Parameter(torch.tensor(.01))
        self.last_choice=None

    @property
    def gamma1(self):
        # Existing H0 monitoring reads this property; it now reports the new
        # local feature residual coefficient, not the removed Stage3 gate.
        return self.gamma

    def forward(self,x,global_context=None,apply_feature_refinement=True,
                disable_skeleton_prediction=False,skeleton_prior=None,
                previous_structure_feat=None,highres_skeleton_feat=None):
        if highres_skeleton_feat is None:raise ValueError('Stage3 needs the live H stream')
        g=self.structure_branch(scale_gradient(x,self.skeleton_gradient_ratio))
        if previous_structure_feat is None:previous_structure_feat=torch.zeros_like(g)
        if previous_structure_feat.shape[-2:]!=g.shape[-2:]:
            previous_structure_feat=F.interpolate(previous_structure_feat,size=g.shape[-2:],mode='bilinear',align_corners=False)
        g=self.previous_structure_fusion(torch.cat((g,previous_structure_feat),1))
        g,h=self.highres_skeleton_fusion(highres_skeleton_feat,g)
        s=self.skeleton_head(h);sp=s.sigmoid()
        direction=self.direction_head(g);d=F.normalize(direction,dim=1,eps=1e-6)
        c=self.connectivity_head(self.connectivity_context(g),skeleton_prob=sp);cp=c.sigmoid()
        hp=self.h_project(h);gp=self.g_project(g);base=torch.cat((hp,gp),1)
        scores=[];neighbours=[];height,width=h.shape[-2:]
        for i,(dy,dx) in enumerate(CONNECTIVITY_DIRECTIONS):
            hn=shifted(hp,dy,dx);gn=shifted(gp,dy,dx);dn=shifted(d,dy,dx)
            norm=float(dy*dy+dx*dx);basis=d.new_tensor([(dx*dx-dy*dy)/norm,2*dx*dy/norm]).view(1,2,1,1)
            geom=h.new_ones((h.shape[0],1,height,width));offset_y=geom*dy;offset_x=geom*dx
            inp=torch.cat((hp,hn,(hp-hn).abs(),hp*hn,gp,gn,sp,shifted(sp,dy,dx),
                cp[:,i:i+1],shifted(cp[:,CONNECTIVITY_OPPOSITE[i]:CONNECTIVITY_OPPOSITE[i]+1],dy,dx),
                (d*basis).sum(1,keepdim=True),(dn*basis).sum(1,keepdim=True),offset_y,offset_x),1)
            valid=shifted(torch.ones_like(sp),dy,dx)>0
            scores.append(self.edge(inp).masked_fill(~valid,torch.finfo(h.dtype).min))
            neighbours.append(shifted(h,dy,dx)-h)
        choice=torch.softmax(torch.cat([self.stay(base)]+scores,1),1)
        delta=sum(choice[:,i+1:i+2]*neighbours[i] for i in range(8))
        output=h+self.gamma*self.out(delta)
        self.last_choice=choice
        info={'structure_feat':g,'highres_skeleton_feat':output,'h2_before_relation':h,
              'relation_weights':choice,'local_delta':output-h}
        return x,s,c,direction,None,info


class NativeGlobalContext(nn.Module):
    def __init__(self, topology, channels=96):
        super().__init__();self.topology=topology
        # The old cosine bias and old surface-gated residual are not used here.
        del self.topology.connectivity_topology_bias_scale
        del self.topology.raw_alpha
        self.to_query=nn.Conv2d(64,channels,1,bias=False)
        self.to_h=nn.Conv2d(channels,64,1,bias=False)
        self.semantic=nn.Linear(channels,16)
        self.pair=nn.Sequential(nn.Linear(58,32),nn.GELU(),nn.Linear(32,topology.heads))
        if topology.alpha_max <= .01:raise ValueError('Global alpha_max must exceed the initial residual .01')
        self.raw_alpha=nn.Parameter(torch.tensor(math.atanh(.01/topology.alpha_max)));self.calls=0

    @property
    def alpha(self):
        return self.topology.alpha_max*torch.tanh(self.raw_alpha)

    def forward(self,h,g,s,c,d):
        b,_,height,width=h.shape;old=self.topology;self.calls+=1
        g=F.interpolate(g,size=(height,width),mode='bilinear',align_corners=False)
        s=F.interpolate(s,size=(height,width),mode='bilinear',align_corners=False).sigmoid()
        c=F.interpolate(c,size=(height,width),mode='bilinear',align_corners=False).sigmoid()
        d=F.normalize(F.interpolate(d,size=(height,width),mode='bilinear',align_corners=False),dim=1,eps=1e-6)
        with torch.no_grad():
            score=.5*old._minmax_normalize_map(h.detach().float().norm(dim=1,keepdim=True))+.5*s.detach()
            coords,valid,_,_=old._extract_fps_anchors(score)
        sample=lambda x:old._sample_features_at_anchor_coords(x,coords,(height,width))
        hs,gs,cs,ds,ss=map(sample,(h,g,c,d,s));k=coords.shape[1]
        nodes=torch.cat((hs,gs,cs,old.node_type_embedding(torch.zeros(b,k,device=h.device,dtype=torch.long)),
                         coords.float()/h.new_tensor([max(height-1,1),max(width-1,1)])),dim=-1)
        nodes=old.node_projection(nodes);sem=self.semantic(gs)
        delta=coords[:,None,:,:].float()-coords[:,:,None,:].float()
        dy,dx=delta[...,0],delta[...,1];dist=(dx.square()+dy.square()).sqrt().clamp_min(1e-6)
        ux,uy=dx/dist,dy/dist;basis=torch.stack((ux.square()-uy.square(),2*ux*uy),-1)
        directions=h.new_tensor([(dx/(dx*dx+dy*dy)**.5,dy/(dx*dx+dy*dy)**.5) for dy,dx in CONNECTIVITY_DIRECTIONS])
        sectors=torch.softmax(4*(ux[...,None]*directions[:,0]+uy[...,None]*directions[:,1]),-1)
        left=(cs[:,:,None,:]*sectors).sum(-1,keepdim=True)
        right=(cs[:,None,:,:]*sectors[...,list(CONNECTIVITY_OPPOSITE)]).sum(-1,keepdim=True)
        di=(ds[:,:,None,:]*basis).sum(-1,keepdim=True);dj=(ds[:,None,:,:]*basis).sum(-1,keepdim=True)
        si=sem[:,:,None,:].expand(-1,-1,k,-1);sj=sem[:,None,:,:].expand(-1,k,-1,-1)
        si_s=ss[:,:,None,:].expand(-1,-1,k,-1);sj_s=ss[:,None,:,:].expand(-1,k,-1,-1)
        rel=torch.cat((si,sj,(si-sj).abs(),left,right,di,dj,si_s,sj_s,
                       ux[...,None],uy[...,None],torch.log1p(dist)[...,None]/math.log1p(max(height,width)),
                       (1/(1+dist))[...,None]),-1)
        qkv=old.token_relation_qkv(nodes).reshape(b,k,3,old.heads,old.channels//old.heads).permute(2,0,3,1,4)
        logits=(qkv[0]@qkv[1].transpose(-2,-1))/math.sqrt(old.channels//old.heads)
        logits=logits+old.token_relation_scale*old._relative_topology_bias(coords,valid,(height,width))+self.pair(rel).permute(0,3,1,2)
        logits=logits.masked_fill(~valid[:,None,None,:],-torch.finfo(logits.dtype).max)
        attended=(torch.softmax(logits,-1)@qkv[2]).transpose(1,2).reshape(b,k,old.channels)
        nodes=(nodes+old.token_relation_projection(attended))*valid.unsqueeze(-1)
        context=old._cross_attention_from_structure_tokens(self.to_query(h),nodes,valid)
        update=self.alpha*self.to_h(old.grid_projection(context))
        self.last_delta_abs_mean=update.detach().abs().mean()
        return h+update


def install_native_relations(model):
    """Switch an already initialized 599a410 H0 into the proposed experiment."""
    core=model.swin_unet
    if not (core.enable_global_topology and core.enable_highres_structure_stream and
            core.enable_h3_surface_fusion and core.stage_skeleton_mode=='direct' and
            core.remove_stage2_pre_topology_source):
        raise ValueError('Native relations require full H0 and direct Stage2/3 skeletons')
    if getattr(core,'native_relation_enabled',False):raise ValueError('Already installed')
    core.decoder_structure_blocks['3']=NativeStage3(core.decoder_structure_blocks['3'])
    core.native_global_context=NativeGlobalContext(core.global_topology)
    core.native_h_surface_proj=nn.Conv2d(64,core.embed_dim,1,bias=False)
    del core.global_topology
    del core.highres_structure_skeleton_head
    core.native_relation_enabled=True
    return model
