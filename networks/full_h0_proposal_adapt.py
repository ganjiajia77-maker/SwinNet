"""Full H0 source transport with an optional unconstrained proposal term.

The old path is retained for same-checkpoint comparisons. Only the multiply
on the positive proposal logit is changed; support remains in both head inputs.
"""
import torch
from torch.nn import functional as F

from networks.full_h0_joint import FullH0Joint
from networks.reliable_directional_repair import directional_visual_sources,compose_surface_logits


class ProposalAdaptedJoint(FullH0Joint):
    proposal_mode='unscaled'

    def forward(self,image,capture_diagnostics=False):
        old=self.original.upstream(image)
        z0,q=old['z0'],old['q']
        highres=old['detail_projected'];size=highres.shape[-2:]
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
        plus=8.*raw[:,:1].sigmoid();minus=4.*raw[:,1:].sigmoid()
        if self.proposal_mode not in ('unscaled','original'):raise ValueError('Invalid proposal mode')
        positive=plus if self.proposal_mode=='unscaled' else positive_support*plus
        proposal=(torch.logit(p0.clamp(1e-5,1.-1e-5))+positive-minus).sigmoid()
        accept_logit=self.original.accept_head(torch.cat((features,proposal-p0),1))
        accept_logit=accept_logit+(bridge-one_side)/(bridge+one_side+.05)
        accept=accept_logit.sigmoid()*(1.-torch.sigmoid(10.*(proposal-p0))*(1.-(.1+.9*positive_support)))
        states,masses,available,logs=self._source_dynamics(visual,p0,q,reliable,proposal,accept,weights)
        z1,delta=compose_surface_logits(z0,p0,states[-1])
        result={'z0':z0,'z1':z1,'p0':z0.sigmoid(),'p1':z1.sigmoid(),'q':q,'q_logits':old['q_logits'],
            'r':reliable,'w':weights,'b':proposal,'a':accept,'u0':states[0],'u1':states[1],'u2':states[2],'u3':states[3],
            'states':states,'selector_context':self._context,'source_available':torch.stack(available,1),
            'source_mass':torch.stack(masses,1),'delta_logit':delta,'delta_z_plus':plus,'delta_z_minus':minus,
            'positive_support':positive_support,'h0_outputs':self.original.upstream.last_outputs,**logs}
        if capture_diagnostics:
            result['feature_shapes']={key:list(old[key].shape) for key in ('detail','semantic','detail_projected','semantic_projected','q_logits')}
        return result
