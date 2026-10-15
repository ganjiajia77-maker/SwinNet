"""Check e2020d9-compatible S/D priors, direction gate behavior and detach rules."""
import ast,hashlib,json,sys
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from networks.skeleton_guided_head import DecoderStructureRefinement
from networks.skeleton_feature_architecture import ARCHITECTURE_ID,require_checkpoint_architecture
REFERENCE_HEAD_AST_SHA256 = 'f3b7a3e580679757509066f0afdcbb2d0974137bbc983a7b24a6e50ca9f150c5'
new=(root/'networks/skeleton_guided_head.py').read_text(encoding='utf-8')
def head_ast(s):
    node=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name=='PairwiseConnectivityHead')
    # Python 3.12 added empty type_params fields; ignore these for a portable hash.
    for child in ast.walk(node):
        child._fields=tuple(field for field in child._fields if field != 'type_params')
    return ast.dump(node,include_attributes=False)
assert hashlib.sha256(head_ast(new).encode()).hexdigest() == REFERENCE_HEAD_AST_SHA256
torch.set_num_threads(2);torch.manual_seed(1234)
block=DecoderStructureRefinement(96,highres_structure_channels=64,use_structure_residual=True).eval()
x=torch.randn(1,96,12,12,requires_grad=True);h=torch.randn(1,64,12,12)
direction=torch.zeros(1,2,12,12);direction[:,0]=1;direction.requires_grad_()
def force_direction(_module,_args,_out):return direction
hook=block.direction_head.register_forward_hook(force_direction)
original_beta=float(block.reliability_beta.detach())
assert original_beta==0
out=block(x,highres_skeleton_feat=h)
grad_d,grad_s=torch.autograd.grad(out[2].sum(),(direction,block.skeleton_head.out.weight),allow_unused=True)
assert grad_d is None and grad_s is None
c_a=out[2].detach()
with torch.no_grad():direction[:,0]=0;direction[:,1]=1
c_b=block(x,highres_skeleton_feat=h)[2].detach()
assert not torch.allclose(c_a,c_b)
block.runtime_connectivity_skeleton_override=torch.zeros(1,1,12,12)
c_zero=block(x,highres_skeleton_feat=h)[2].detach()
block.runtime_connectivity_skeleton_override=torch.ones(1,1,12,12)
c_one=block(x,highres_skeleton_feat=h)[2].detach()
assert not torch.allclose(c_zero,c_one)
block.runtime_connectivity_skeleton_override=None
block.runtime_ablate_connectivity_gate=True
block.runtime_ablate_skeleton_prior=True
with torch.no_grad():block.reliability_beta.fill_(.25)
base=block(x,highres_skeleton_feat=h)
with torch.no_grad():direction.mul_(3)
scaled=block(x,highres_skeleton_feat=h)
torch.testing.assert_close(base[2],scaled[2])
torch.testing.assert_close(base[5]['structure_gate_old'],scaled[5]['structure_gate_old'])
assert not torch.allclose(base[4],scaled[4])
expected=torch.sigmoid(torch.logit(scaled[5]['structure_gate_old'])+block.reliability_beta*scaled[5]['reliability_correction'])
torch.testing.assert_close(scaled[4],expected)
grad_d=torch.autograd.grad(scaled[4].sum(),direction,allow_unused=True,retain_graph=True)[0]
assert grad_d is None
scaled[4].mean().backward()
assert sum(float(p.grad.abs().sum()) for p in block.reliability_correction.parameters() if p.grad is not None)>0
with torch.no_grad():block.reliability_beta.zero_()
zero_beta=block(x,highres_skeleton_feat=h)
torch.testing.assert_close(zero_beta[4],zero_beta[5]['structure_gate_old'],rtol=0,atol=0)
hook.remove()
try:require_checkpoint_architecture({'skeleton_feature_architecture':{'id':'h0_skeleton_feature_stream_v1','global_topology':False}})
except ValueError:pass
else:raise AssertionError('Old v1 checkpoint was accepted')
report={'architecture_id':ARCHITECTURE_ID,'reference_commit':'e2020d9a8df311f9f680cd3637466a674d76688e','connectivity_head_AST_identical_to_reference':True,'S_changes_C':True,'D_alignment_changes_C':True,'C_does_not_backpropagate_through_S_or_D_prior':True,'D_norm_changes_gate_when_beta_nonzero':True,'D_to_gate_confidence_is_detached':True,'gate_formula_matches_reference':True,'reliability_beta_initial':original_beta,'zero_beta_gate_equals_old_gate':True,'correction_module_has_gradient_when_beta_nonzero':True,'v1_checkpoint_rejected':True}
(root/'skeleton_feature_sdgate_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
