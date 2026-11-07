"""Focused x4 readout/expansion checks and frozen A1/A2 interventions."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time
from collections import Counter
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_native_h0_pair import load_model,digest
from diagnose_b_stage_features import model_digest,count,metrics,csv_out,dump
from diagnostics.stage_feature_readout import LinearReadout,road_loss
from diagnostics.sampling_audit import (UpsampleCapture,PhaseReadout,expansion_matrix_report,
    reconstruction_report,prior_intervention,prior_parameters)
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from mass_protocol import sliding_positions,blend_weight
from evaluate_local_cs import local_closed
from selector_transport_reference import bridge_proxy


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source_dir',type=Path,required=True,help='Existing b_stage_features output')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--root_path',type=Path,required=True)
    p.add_argument('--fixed_gaps',type=Path,required=True)
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--extra_probe_epochs',type=int,default=2)
    p.add_argument('--probe_lr',type=float,default=.003)
    p.add_argument('--skip_priors',action='store_true')
    a=p.parse_args();out=a.output_dir
    if out.exists():raise ValueError('Use a new output directory')
    if a.extra_probe_epochs<1:raise ValueError('Need at least one probe epoch')
    out.mkdir(parents=True)
    t0=time.perf_counter()
    source=torch.load(a.source_dir/'readouts.pth',map_location='cpu',weights_only=False)
    if source['checkpoint_sha256']!=digest(a.checkpoint):raise ValueError('Checkpoint differs from original feature diagnostic')
    settings=source['args'];split=json.loads((a.source_dir/'split.json').read_text())
    torch.manual_seed(settings['seed']);np.random.seed(settings['seed'])
    model,saved=load_model(a.checkpoint,torch.device('cuda'),True)
    model.eval().requires_grad_(False);before_hash=model_digest(model);core=model.swin_unet
    matrix=expansion_matrix_report(core.up)
    dump(out/'x4_matrix.json',matrix)
    print('[EXPANSION MATRIX]',json.dumps(matrix),flush=True)
    heads={}
    for short,name in [('before_x4','decoder_final_norm'),('old_after_x4','final_x4_after_up')]:
        head=LinearReadout(core.embed_dim)
        head.load_state_dict({k[len(name)+1:]:v for k,v in source['probes'].items() if k.startswith(name+'.')},strict=True)
        heads[short]=head.cuda().requires_grad_(False)
    heads['refit_shared']=copy.deepcopy(heads['old_after_x4']).requires_grad_(True)
    heads['refit_phase']=PhaseReadout(heads['old_after_x4']).cuda()
    dump(out/'probe_config.json',dict(extra_epochs=a.extra_probe_epochs,lr=a.probe_lr,optimizer='fresh Adam',
        normalization='reuse original training-only statistics',phase_initialization='all phases equal original shared head',
        head_parameters={n:sum(p.numel() for p in h.parameters()) for n,h in heads.items()},
        note='Phase readout has 16x parameters. Improvement alone is not proof of irreversible information loss.'))
    capture=UpsampleCapture(model)
    training=RoadSkeletonDataset(str(a.root_path),'train',image_size=512,source_patch_size=1500,
        random_crop_train=True,random_crops_per_image=settings['crops_per_image'],
        random_crop_seed=settings['seed'],augment=False)
    image_index={name:i for i,name in enumerate(training.image_files)}
    crop_indices=lambda names:[image_index[n]*settings['crops_per_image']+j for n in names for j in range(settings['crops_per_image'])]
    fit=crop_indices(split['fit_images']);held=crop_indices(split['holdout_images'])
    opt=torch.optim.Adam(list(heads['refit_shared'].parameters())+list(heads['refit_phase'].parameters()),lr=a.probe_lr)
    losses=[]
    for epoch in range(a.extra_probe_epochs):
        for j,index in enumerate(fit):
            sample=training[index]
            with torch.no_grad():model(sample['image'][None].cuda())
            if epoch==0 and j==0:
                reconstruction=reconstruction_report(core.up,capture.values)
                dump(out/'x4_reconstruction.json',reconstruction)
                print('[EXPANSION RECONSTRUCTION]',json.dumps(reconstruction),flush=True)
            target=sample['mask'][None].cuda();opt.zero_grad(set_to_none=True)
            current={n:road_loss(heads[n](capture.values['post']),target) for n in ('refit_shared','refit_phase')}
            sum(current.values()).backward();opt.step()
            if any(p.grad is not None for p in model.parameters()):raise RuntimeError('B received gradients')
            losses.append(dict(epoch=epoch+1,crop=j+1,**{n:float(v.detach()) for n,v in current.items()}))
            if j%20==0 or j+1==len(fit):print(f'[READOUT CONTINUE] {epoch+1}/{a.extra_probe_epochs} {j+1}/{len(fit)}',flush=True)
    csv_out(out/'continued_probe_losses.csv',losses)
    ts=np.round(np.arange(.01,.991,.01),2)
    totals={n:np.zeros((len(ts),3),np.int64) for n in heads};hold_losses=Counter()
    with torch.no_grad():
        for j,index in enumerate(held):
            sample=training[index];model(sample['image'][None].cuda());gt=sample['mask'][0].numpy()>.5
            for n,h in heads.items():
                z=h(capture.values['parent'],(512,512)) if n=='before_x4' else h(capture.values['post'])
                hold_losses[n]+=float(road_loss(z,sample['mask'][None].cuda()))
                prob=z.sigmoid()[0,0].cpu().numpy()
                for k,t in enumerate(ts):totals[n][k]+=list(count(prob,gt,t).values())
            print(f'[HOLDOUT] {j+1}/{len(held)}',flush=True)
    thresholds={};hold_rows=[]
    for n in heads:
        rows=[dict(head=n,threshold=float(t),**metrics(dict(zip(('tp','fp','fn'),map(int,c))))) for t,c in zip(ts,totals[n])]
        hold_rows.extend(rows);thresholds[n]=max(rows,key=lambda r:(r['iou'],-r['threshold']))['threshold']
    csv_out(out/'holdout_thresholds.csv',hold_rows)
    dump(out/'holdout_losses.json',{n:value/len(held) for n,value in hold_losses.items()})
    dump(out/'thresholds.json',dict(selected_on='same_train_image_holdout',thresholds=thresholds))
    merge,bias=prior_parameters(core)
    dump(out/'A1A2_parameters.json',dict(
        merge_road_alpha=[float(p.detach()) for p in merge],stage3_bias_scales=[float(p.detach()) for p in bias],
        merge_modulated_layers=[i+1 for i,layer in enumerate(core.layers) if layer.road_attention_modulates_downsample],
        merge_modes=[core.layers[i].downsample.road_attention_merge_mode for i in (0,1)],
        third_merge_has_direct_A1A2=False,
        notes='Frozen interventions measure checkpoint dependence, not training-time module ablation.'))
    modes=['merge12_off','stage3_bias_off','both_off'] if not a.skip_priors else []
    validation=RoadSkeletonDataset(str(a.root_path),'val',image_size=None,source_patch_size=1500,return_full_image=True)
    reference=json.loads(a.fixed_gaps.read_text());rows=[];gaps=[];fixed_threshold=float(settings['threshold'])
    blend=blend_weight(512,torch.device('cuda'))[0,0].cpu().numpy()
    with torch.no_grad():
        for i in range(len(validation)):
            sample=validation[i];case=sample['case_name'];gt=sample['mask'][0].numpy()>.5
            if gt.shape!=(1500,1500) or case not in reference['images']:raise ValueError('Reference mismatch')
            image=sample['image'][None].cuda()
            canvases={n:np.zeros(gt.shape,np.float32) for n in list(heads)+['actual_B']+modes};divisor=np.zeros(gt.shape,np.float32)
            for top in sliding_positions(1500,512,256):
                for left in sliding_positions(1500,512,256):
                    tile=image[:,:,top:top+512,left:left+512];sl=np.s_[top:top+512,left:left+512]
                    actual=model(tile)[0];canvases['actual_B'][sl]+=actual[0,0].cpu().numpy()*blend
                    for n,h in heads.items():
                        z=h(capture.values['parent'],(512,512)) if n=='before_x4' else h(capture.values['post'])
                        canvases[n][sl]+=z[0,0].cpu().numpy()*blend
                    for mode in modes:
                        with prior_intervention(core,mode):z=model(tile)[0]
                        canvases[mode][sl]+=z[0,0].cpu().numpy()*blend
                    divisor[sl]+=blend
            probabilities={n:1/(1+np.exp(-np.clip(v/divisor,-80,80))) for n,v in canvases.items()}
            baseline=probabilities['actual_B']>=fixed_threshold
            for n,prob in probabilities.items():
                selections={'fixed':fixed_threshold}
                if n in thresholds:selections['own_train_holdout']=thresholds[n]
                for setting,t in selections.items():
                    pred=prob>=t;c=count(prob,gt,t)
                    rows.append(dict(case=case,variant=n,setting=setting,threshold=t,**c,**metrics(c),
                        added_tp=int((pred & ~baseline & gt).sum()),lost_tp=int((~pred & baseline & gt).sum()),
                        added_fp=int((pred&~baseline&~gt).sum()),removed_fp=int((~pred&baseline&~gt).sum()),
                        new_bridge_proxy=bridge_proxy(baseline,pred,gt)))
                    for gap in reference['images'][case]['gaps']:
                        old=local_closed(baseline,gap);closed=local_closed(pred,gap)
                        gaps.append(dict(case=case,gap_id=gap['id'],variant=n,setting=setting,closed=closed,
                                         initial_closed=old,newly_closed=closed and not old,lost=old and not closed))
            print(f'[VALIDATION] {i+1}/{len(validation)} {case}',flush=True)
    csv_out(out/'per_image.csv',rows);csv_out(out/'gap_changes.csv',gaps)
    summary=[]
    for n in list(heads)+['actual_B']+modes:
        for setting in ('fixed','own_train_holdout'):
            selected=[r for r in rows if r['variant']==n and r['setting']==setting]
            if not selected:continue
            c={k:sum(r[k] for r in selected) for k in ('tp','fp','fn','added_tp','lost_tp','added_fp','removed_fp','new_bridge_proxy')}
            g=[r for r in gaps if r['variant']==n and r['setting']==setting]
            summary.append(dict(variant=n,setting=setting,threshold=selected[0]['threshold'],**metrics(c),
                **{k:v for k,v in c.items() if k not in ('tp','fp','fn')},
                closed=sum(r['closed'] for r in g),newly_closed=sum(r['newly_closed'] for r in g),lost=sum(r['lost'] for r in g)))
    csv_out(out/'comparison.csv',summary)
    capture.close();after_hash=model_digest(model)
    if before_hash!=after_hash:raise RuntimeError('B parameters/buffers changed or intervention not restored')
    dump(out/'checks.json',dict(model_unchanged=True,before=before_hash,after=after_hash,
        B_has_grad=any(p.grad is not None for p in model.parameters()),elapsed_seconds=time.perf_counter()-t0,
        source_dir=str(a.source_dir),checkpoint_sha256=digest(a.checkpoint)))
    (out/'REPORT.md').write_text(
        '# 原B采样与读出核查\n\n'
        'x4_matrix.json：线性扩展的列秩；满列秩意味着线性扩展加重排在LayerNorm之前没有丢失输入向量信息。\n\n'
        'x4_reconstruction.json：实际特征的数值逆重建。LayerNorm重建额外使用保存的均值和标准差，仅用于核查，不是假设普通读出头具备这些信息。\n\n'
        'comparison.csv：旧读出、继续训练的共享读出、16相位读出，以及A1/A2实际输出干预。相位头参数更多，单凭其改善不能宣称原上采样有错。\n\n'
        'A1A2_parameters.json：前两次merge调制与Stage3偏置的实际参数；最大的第三次merge不直接使用A1/A2。\n\n'
        '关闭先验后变差支持当前模型依赖这条路径；关闭后改善提示当前使用值得检查。它们均不能替代从相同初始化重新训练的模块消融。\n\n'
        '阈值由原训练图像留出集选取，验证集不训练读出。固定阈值比较原模型干预；缺口闭合与桥接是既有代理。\n',encoding='utf-8')
    print('[ALL DONE]',out,flush=True)


if __name__=='__main__':main()
