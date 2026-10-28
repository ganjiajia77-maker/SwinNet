"""Validation-only local greedy GT gate intervention, never a deployed model."""
import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
import run_reliable_v9_mass as v9
from mass_protocol import blend_weight, sliding_positions
from networks.reliable_directional_repair import DIRECTIONS, shifted
from diagnostics.diagnose_v9_transport import matched_threshold
from diagnostics.v91_counterfactual import evaluate, write_json


def oracle_choice(state, proposal, anchor, transport, target, original_gate,
                  min_effect=1e-5, min_gain=1e-7):
    """Pick G=0/1 by immediate per-cell BCE, keeping original G if uncertain.

    `transport` is the pre-G transport_rate*(M-U). Proposal and anchor are
    already signed increments. This does not optimize final path connectivity.
    """
    gt = F.interpolate(target.float(), size=state.shape[-2:], mode='area')
    valid = (gt >= .75) | (gt <= .05)
    hard_gt = (gt >= .75).to(state.dtype)
    baseline = (state + proposal + anchor).clamp(1e-6, 1.-1e-6)
    proposed = (state + proposal + anchor + transport).clamp(1e-6, 1.-1e-6)
    loss0 = F.binary_cross_entropy(baseline, hard_gt, reduction='none')
    loss1 = F.binary_cross_entropy(proposed, hard_gt, reduction='none')
    gain = loss0 - loss1
    decisive = valid & (transport.abs() >= min_effect) & (gain.abs() >= min_gain)
    chosen = torch.where(decisive, (gain > 0).to(state.dtype), original_gate)
    stats = {'allowed': int((decisive & (gain > 0)).sum()),
             'blocked': int((decisive & (gain < 0)).sum()),
             'uncertain': int((~decisive).sum()),
             'mixed_gt': int((~valid).sum()),
             'tiny_action': int((valid & (transport.abs() < min_effect)).sum()),
             'near_tie': int((valid & (transport.abs() >= min_effect) & (gain.abs() < min_gain)).sum()),
             'positive_action': int((decisive & (transport > 0)).sum()),
             'negative_action': int((decisive & (transport < 0)).sum())}
    for region, region_mask in (('road',gt>=.75),('background',gt<=.05)):
        for direction, direction_mask in (('positive',transport>0),('negative',transport<0)):
            for decision,decision_mask in (('allow',gain>0),('block',gain<0)):
                stats[f'{region}_{direction}_{decision}'] = int((decisive & region_mask & direction_mask & decision_mask).sum())
    return chosen, stats


@contextmanager
def oracle_dynamics(module, target, counts, min_effect=1e-5, min_gain=1e-7):
    original = module.forward
    def intervention(self, visual, p0, q, reliable, proposal, accept, weights,
                     capture_diagnostics=False, transport_gate=None,
                     freeze_generated_sources=False, disable_transport=False,
                     disable_proposal=False, state_gate=None, base_gate_logit=None):
        if transport_gate is None or freeze_generated_sources or disable_transport or disable_proposal:
            raise ValueError('Oracle audit expects the ordinary V9 transport path')
        state = p0
        states, masses, available_list = [state], [], []
        details = {k: [] for k in ('alpha','beta','gamma','stay','m','transport_delta',
                                   'proposal_delta','anchor_delta','g_step','state_gate_delta','transport_rate')}
        for step in range(self.steps):
            gained = ((state-p0).clamp_min(0.)/(1.-p0).clamp_min(1e-5)).clamp_max(1.)
            support = reliable + (1.-reliable)*gained
            mass = torch.zeros_like(state); numerator = torch.zeros_like(state)
            for idx, (dy, dx) in enumerate(DIRECTIONS):
                src = weights[:, idx:idx+1]*shifted(support, dy, dx)
                mass = mass + src; numerator = numerator + src*shifted(state, dy, dx)
            available = mass.clamp(0., 1.)
            propagated = torch.where(mass > 1e-6, numerator/mass.clamp_min(1e-6), state)
            rates = self.rates(torch.cat((visual,p0,q,reliable,proposal,accept,state,available),dim=1)).softmax(dim=1)
            delta = torch.zeros_like(state)
            old_gate = transport_gate
            if state_gate is not None:
                delta = state_gate(torch.cat((state,propagated-state,available,rates[:,1:2],
                                              proposal-state,p0-state),dim=1))
                old_gate = torch.sigmoid(base_gate_logit + delta)
            beta = .2*accept*rates[:,2:3]; gamma = rates[:,3:4]
            potential = rates[:,1:2]*(propagated-state)
            chosen, row = oracle_choice(state, beta*(proposal-state), gamma*(p0-state),
                                        potential, target, old_gate, min_effect, min_gain)
            for key, value in row.items(): counts[step][key] = counts[step].get(key,0) + value
            alpha = rates[:,1:2]*chosen; stay = 1.-alpha-beta-gamma
            if capture_diagnostics:
                for key,value in (('alpha',alpha),('beta',beta),('gamma',gamma),('stay',stay),
                    ('m',propagated),('transport_delta',alpha*(propagated-state)),
                    ('proposal_delta',beta*(proposal-state)),('anchor_delta',gamma*(p0-state)),
                    ('g_step',chosen),('state_gate_delta',delta),('transport_rate',rates[:,1:2])):
                    details[key].append(value)
            state = stay*state + alpha*propagated + beta*proposal + gamma*p0
            states.append(state); masses.append(mass); available_list.append(available)
        if capture_diagnostics:
            return states, masses, available_list, {k: torch.stack(v, dim=1) for k,v in details.items()}
        return states, masses, available_list
    module.forward = MethodType(intervention, module)
    try:
        yield
    finally:
        module.forward = original


@torch.no_grad()
def infer(model, dataset, device, counts, min_effect, min_gain):
    cases = []
    for index, sample in enumerate(dataset):
        image = sample['image'][None].to(device); mask = sample['mask'][None].to(device)
        gt = sample['mask'][0].numpy() > .5; h,w = gt.shape
        canvas = {k: torch.zeros((1,1,h,w), device=device) for k in ('p0','A_original','O_oracle')}
        divisor = torch.zeros_like(canvas['p0']); weight = blend_weight(512, device)
        for y in sliding_positions(h,512,256):
            for x in sliding_positions(w,512,256):
                frozen = model.frozen(image[:,:,y:y+512,x:x+512], capture_diagnostics=True)
                normal = model(None, frozen_override=frozen)
                with oracle_dynamics(model.dynamics, mask[:,:,y:y+512,x:x+512], counts, min_effect, min_gain):
                    oracle = model(None, frozen_override=frozen)
                if not torch.equal(normal['z0'], oracle['z0']): raise RuntimeError('Oracle changed P0')
                region = (slice(None),slice(None),slice(y,y+512),slice(x,x+512))
                for name, value in (('p0',normal['z0']),('A_original',normal['z1']),('O_oracle',oracle['z1'])):
                    canvas[name][region] += value.float()*weight
                divisor[region] += weight
        if not bool((divisor > 0).all()): raise ValueError('Uncovered image pixels')
        logits = {name: (value/divisor)[0,0].cpu().numpy() for name,value in canvas.items()}
        probs = {name: 1/(1+np.exp(-np.clip(value,-80,80))) for name,value in logits.items()}
        if any(not np.isfinite(v).all() for v in probs.values()): raise ValueError('Nonfinite output')
        cases.append({'case':sample['case_name'], 'gt':gt, **probs})
        print(f'[ORACLE] {index+1}/{len(dataset)} {sample["case_name"]}',flush=True)
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir',type=Path,required=True); p.add_argument('--root_path',type=Path,required=True)
    p.add_argument('--reference_set',type=Path,required=True); p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--min_effect',type=float,default=1e-5)
    p.add_argument('--min_gain',type=float,default=1e-7); p.add_argument('--matching_bins',type=int,default=10000)
    args=p.parse_args()
    if args.min_effect <= 0 or args.min_gain <= 0 or args.matching_bins < 1000: p.error('Invalid diagnostic settings')
    ckpt=args.run_dir/'stage_D/best.pth'; selected_path=args.run_dir/'direct_val_selection.json'
    if not ckpt.is_file() or not selected_path.is_file() or not args.reference_set.is_file(): p.error('Missing V9 checkpoint, val selection or V9.1 reference_set.json')
    if args.output_dir.exists() and any(args.output_dir.iterdir()): p.error('Use an empty output directory')
    payload=torch.load(ckpt,map_location='cpu',weights_only=False); selection=json.loads(selected_path.read_text())
    if payload['identity'] != selection['identity'] or selection.get('selected_on') != 'val' or Path(selection['checkpoint']).resolve() != ckpt.resolve(): p.error('Checkpoint and val selection differ')
    if Path(payload['training_config']['root_path']).resolve() != args.root_path.resolve(): p.error('Dataset root differs from V9')
    reference=float(payload['training_config']['reference_threshold'])
    if float(selection['report']['p0_reference_threshold']) != reference or selection['report']['paired']['eligible_p0_gaps'] != 206: p.error('Expected the fixed 206-gap P0 reference')
    checksum=v9.digest(ckpt); device=torch.device(args.device)
    model,_,identity=v9.load_initial(SimpleNamespace(initialization='direct',q3_checkpoint=Path(payload['identity']['q3']['path']),device=device))
    v9.restore_checkpoint(model,ckpt,identity,'D'); model.eval()
    from run_q3_r3_mass import full_dataset
    dataset=full_dataset(args.root_path,'val')
    if len(dataset) != 14: p.error('Expected 14 val images')
    counts=[dict(allowed=0,blocked=0,uncertain=0,positive_action=0,negative_action=0) for _ in range(3)]
    cases=infer(model,dataset,device,counts,args.min_effect,args.min_gain)
    threshold=float(selection['threshold']); saved=selection['report']
    for field, key in (('p0','p0'),('A_original','p1')):
        totals={'tp':sum(int(((c[field]>=threshold)&c['gt']).sum()) for c in cases),
                'fp':sum(int(((c[field]>=threshold)&~c['gt']).sum()) for c in cases),
                'fn':sum(int(((c[field]<threshold)&c['gt']).sum()) for c in cases)}
        if any(totals[k] != saved[key][k] for k in totals): raise ValueError(f'Original {field} counts do not reproduce saved val selection: {totals}')
    target_tp=sum(int(((c['A_original']>=threshold)&c['gt']).sum()) for c in cases)
    target_area=sum(int((c['A_original']>=threshold).sum()) for c in cases)
    points={'A_original':{'same_threshold':threshold,'matched_recall':threshold,'matched_foreground_area':threshold},
            'O_oracle':{'same_threshold':threshold,
                        'matched_recall':matched_threshold(cases,'O_oracle',target_tp,'tp',args.matching_bins)[0],
                        'matched_foreground_area':matched_threshold(cases,'O_oracle',target_area,'area',args.matching_bins)[0]}}
    # This additional fixed operating point was specified from earlier val work.
    for mode in points: points[mode]['fixed_0_35'] = .35
    args.output_dir.mkdir(parents=True,exist_ok=True)
    reports=evaluate(cases,points,reference,206,args.output_dir,modes=('A_original','O_oracle'))
    if json.loads((args.output_dir/'reference_set.json').read_text()) != json.loads(args.reference_set.read_text()):
        raise ValueError('The 206-gap P0/GT fingerprint differs from V9.1')
    if v9.digest(ckpt)!=checksum: raise RuntimeError('V9 checkpoint changed')
    result={'checkpoint_sha256':checksum,'selected_on':'val','images':14,'counts_per_window_step':counts,
            'comparisons':reports,'method':'Greedy per-step GT BCE choice; ambiguous/mixed or tiny actions retain original G. No change to network weights. Rates and sources recompute on intervened states.',
            'limitations':'GT oracle is nondeployable; local BCE greedy choice is not a mathematical upper bound on global path metrics. Sliding-window counts overlap. Metrics are proxies.'}
    write_json(args.output_dir/'oracle_gate.json',result)
    lines=['# V9 local GT oracle gate (validation only)','','| mode | point | threshold | IoU | clDice | bidirectional APLS | closed / 206 | background bridge proxy | short components |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in points:
        for point,row in reports[name].items():
            m=row['metrics']; b=row['paired']
            lines.append(f'| {name} | {point} | {row["threshold"]:.4f} | {m["iou"]:.4f} | {m["cldice"]:.4f} | {m["apls_bidirectional_approx"]:.4f} | {b["closed"]}/206 | {b["background_bridge_proxy"]} | {m["short_pred_components"]:.2f} |')
    lines+=['','GT is used only in this oracle intervention. It is a local greedy diagnostic, not a global upper bound or deployable score. Matching residuals and length bins are in oracle_gate.json.']
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'[DONE] {args.output_dir}',flush=True)

if __name__=='__main__': main()
