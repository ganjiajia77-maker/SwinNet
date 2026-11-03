"""Only the existing unscaled proposal head is trained in this experiment."""
import torch
from torch.nn import functional as F
from networks.full_h0_proposal_adapt import ProposalAdaptedJoint


class DirectCandidateJoint(ProposalAdaptedJoint):
    def forward(self,image,capture_diagnostics=False):
        captured=[]
        handle=self.original.proposal_head.register_forward_hook(lambda module,inputs,output:captured.append((inputs[0],output)))
        try:result=super().forward(image,capture_diagnostics)
        finally:handle.remove()
        if self.proposal_mode!='unscaled' or len(captured)!=1:raise RuntimeError('Expected one unscaled proposal forward')
        features,raw=captured[0]
        # This is the raw expression used by the head, never reconstructed from B.
        logit=torch.logit(result['u0'].clamp(1e-5,1.-1e-5))+8.*raw[:,:1].sigmoid()-4.*raw[:,1:].sigmoid()
        if not torch.allclose(logit.sigmoid(),result['b'],rtol=1e-6,atol=1e-7):raise RuntimeError('B/logit mismatch')
        result['b_logit']=logit
        if capture_diagnostics:result['proposal_input']=features
        return result


def freeze_except_proposal(model):
    model.requires_grad_(False);model.original.proposal_head.requires_grad_(True);model.eval()
    names=[name for name,p in model.named_parameters() if p.requires_grad]
    if not names or any(not name.startswith('original.proposal_head.') for name in names):
        raise RuntimeError('Only proposal_head may train')
    return names


def candidate_segmentation_loss(result,mask,skeleton,lambda_bg=1.,lambda_missing=2.,missing_threshold=.45):
    """Exactly dynamic_losses segmentation; B logits are upsampled before sigmoid."""
    logits=F.interpolate(result['b_logit'],size=mask.shape[-2:],mode='bilinear',align_corners=False)
    probability=logits.sigmoid();p0=result['p0'];q=result['q']
    if p0.shape!=mask.shape or skeleton.shape!=mask.shape:raise ValueError('GT and P0 sizes differ')
    q_full=F.interpolate(q.detach(),size=mask.shape[-2:],mode='bilinear',align_corners=False)
    missed=(skeleton>.5)&(F.max_pool2d(p0.detach(),5,stride=1,padding=2)<missing_threshold)
    band=F.max_pool2d(missed.float(),5,stride=1,padding=2)
    weights=1.+lambda_bg*(1.-mask)*q_full+lambda_missing*mask*band
    bce=F.binary_cross_entropy_with_logits(logits,mask,reduction='none')
    focal=(.75*(1.-torch.exp(-bce)).square()*bce*weights).sum()/weights.sum().clamp_min(1.)
    dice=1.-(2.*(probability*mask).sum((1,2,3))+1.)/(
        probability.sum((1,2,3))+mask.sum((1,2,3))+1.)
    loss=focal+.5*dice.mean()
    return loss,{'focal':focal.detach(),'dice':dice.mean().detach(),
        'missed_skeleton_pixels':int(missed.sum()),'weighted_background_pixels':int(((mask<.5)&(q_full>.5)).sum())}
