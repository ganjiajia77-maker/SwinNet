"""Locate changes in decodable road information across every native B stage.

Only independent 1x1 readout heads are fitted. Native B stays frozen/eval.
"""
import argparse
from collections import Counter
import csv
import html
import json
import math
from pathlib import Path
import random
import sys
import time
import cv2
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_native_h0_pair import load_model, digest
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diagnostics.stage_feature_readout import StageCapture, LinearReadout, road_loss
from mass_protocol import sliding_positions, blend_weight
from evaluate_local_cs import local_closed


def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')


def csv_out(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def model_digest(model):
    import hashlib
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def count(prob,gt,t):
    pred=prob>=t
    return dict(tp=int((pred & gt).sum()),fp=int((pred & ~gt).sum()),fn=int((~pred & gt).sum()))


def metrics(c):
    tp,fp,fn=c['tp'],c['fp'],c['fn']
    return dict(iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
                f1=2*tp/max(2*tp+fp+fn,1))


def rgb(sample):
    x=sample['image'].numpy().transpose(1,2,0)
    return np.clip((x*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255,0,255).astype(np.uint8)


def write_image(path,array):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),array):raise IOError(path)


def pick_panel(image,title):
    panel=cv2.resize(image,(220,220),interpolation=cv2.INTER_NEAREST)
    panel=cv2.copyMakeBorder(panel,36,0,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
    # OpenCV labels deliberately ASCII; complete stage descriptions in HTML/CSV.
    cv2.putText(panel,title[:31],(4,17),cv2.FONT_HERSHEY_SIMPLEX,.37,(0,0,0),1,cv2.LINE_AA)
    return panel


def gallery(out,case,image,gt,reference,names,limit):
    if limit<=0:return []
    rows=[]
    for gap in reference['gaps'][:limit]:
        yy,xx=np.divmod(np.array(gap['pixels']),gt.shape[1]);cy,cx=int(yy.mean()),int(xx.mean())
        y0=max(0,min(gt.shape[0]-160,cy-80));x0=max(0,min(gt.shape[1]-160,cx-80))
        sl=np.s_[y0:y0+160,x0:x0+160]
        panels=[pick_panel(cv2.cvtColor(image[sl],cv2.COLOR_RGB2BGR),'RGB'),
                pick_panel(cv2.cvtColor((gt[sl]*255).astype(np.uint8),cv2.COLOR_GRAY2BGR),'GT road')]
        for name in names:
            prob=cv2.imread(str(out/'predictions'/name/(case+'.png')),cv2.IMREAD_GRAYSCALE)
            color=cv2.applyColorMap(prob[sl],cv2.COLORMAP_TURBO)
            contours,_=cv2.findContours(gt[sl].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(color,contours,-1,(255,255,255),1)
            panels.append(pick_panel(color,name))
        while len(panels)%5:panels.append(np.full_like(panels[0],255))
        sheet=np.concatenate([np.concatenate(panels[i:i+5],axis=1) for i in range(0,len(panels),5)],axis=0)
        name=f'{case}_gap_{gap["id"]}.jpg';write_image(out/'cases'/name,sheet)
        rows.append(dict(case=case,gap_id=gap['id'],file='cases/'+name,top=y0,left=x0))
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--root_path',type=Path,required=True)
    p.add_argument('--fixed_gaps',type=Path,required=True)
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--train_images',type=int,default=64)
    p.add_argument('--holdout_images',type=int,default=16)
    p.add_argument('--crops_per_image',type=int,default=2)
    p.add_argument('--probe_epochs',type=int,default=3)
    p.add_argument('--normalization_crops',type=int,default=32)
    p.add_argument('--probe_lr',type=float,default=.01)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--threshold',type=float,default=.2)
    p.add_argument('--case_panels',type=int,default=24)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if min(a.train_images,a.holdout_images,a.crops_per_image,a.probe_epochs,a.normalization_crops)<=0:
        raise ValueError('Counts must be positive')
    if a.output_dir.exists():raise ValueError('Use a new output directory')
    out=a.output_dir;out.mkdir(parents=True)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; choose an available GPU')
    model,saved=load_model(a.checkpoint,device,True)
    if saved.get('path_training_mode','off')!='off' or saved.get('local_cs_mode','off')!='off':
        raise ValueError('Use original native B checkpoint for this diagnostic')
    model.requires_grad_(False);model.eval()
    original_digest=model_digest(model)
    training=RoadSkeletonDataset(str(a.root_path),'train',image_size=512,source_patch_size=1500,
        random_crop_train=True,random_crops_per_image=a.crops_per_image,random_crop_seed=a.seed,augment=False)
    order=np.random.default_rng(a.seed).permutation(len(training.image_files))
    if a.train_images+a.holdout_images>len(order):raise ValueError('Not enough distinct training images')
    fit_images=order[:a.train_images].tolist();held_images=order[a.train_images:a.train_images+a.holdout_images].tolist()
    indices=lambda images:[i*a.crops_per_image+j for i in images for j in range(a.crops_per_image)]
    fit=indices(fit_images);held=indices(held_images)
    dump(out/'split.json',dict(seed=a.seed,fit_images=[training.image_files[i] for i in fit_images],
        holdout_images=[training.image_files[i] for i in held_images],fit_crops=len(fit),holdout_crops=len(held),
        augment=False,crops_fixed_across_probe_epochs=True))
    capture=StageCapture(model)
    t0=time.perf_counter()
    first=training[fit[0]]['image'][None].to(device)
    with torch.no_grad():
        capture.active=False;without=model(first)[0].clone()
        capture.active=True;capture.clear();with_hooks=model(first)[0]
        hook_delta=float((without-with_hooks).abs().max())
    if hook_delta!=0:raise RuntimeError('Hooks changed B output')
    names=list(capture.values)
    probes=nn.ModuleDict({name:LinearReadout(capture.values[name].shape[1]) for name in names}).to(device)
    stage_rows=[dict(index=i,name=name,channels=capture.metadata[name][0],height=capture.metadata[name][1],
        width=capture.metadata[name][2],readout_parameters=sum(v.numel() for v in probes[name].parameters())) for i,name in enumerate(names)]
    csv_out(out/'stages.csv',stage_rows)
    print('[STAGES]',json.dumps(stage_rows),flush=True)
    # Normalization statistics are fitted exclusively on training images.
    sums={name:None for name in names};squares={name:None for name in names};pixels=Counter()
    with torch.no_grad():
        for j,index in enumerate(fit[:a.normalization_crops]):
            capture.clear();model(training[index]['image'][None].to(device))
            for name,x in capture.values.items():
                mean=x.mean((0,2,3));second=x.square().mean((0,2,3));n=x.shape[0]*x.shape[2]*x.shape[3]
                sums[name]=mean*n if sums[name] is None else sums[name]+mean*n
                squares[name]=second*n if squares[name] is None else squares[name]+second*n
                pixels[name]+=n
            print(f'[NORMALIZE] {j+1}/{min(len(fit),a.normalization_crops)}',flush=True)
        for name in names:
            mean=sums[name]/pixels[name];std=(squares[name]/pixels[name]-mean.square()).clamp_min(1e-6).sqrt()
            probes[name].mean.copy_(mean[None,:,None,None]);probes[name].std.copy_(std[None,:,None,None])
    optimizer=torch.optim.Adam(probes.parameters(),lr=a.probe_lr)
    loss_rows=[]
    for epoch in range(a.probe_epochs):
        for j,index in enumerate(fit):
            sample=training[index];capture.clear()
            with torch.no_grad():model(sample['image'][None].to(device))
            target=sample['mask'][None].to(device);optimizer.zero_grad(set_to_none=True)
            losses={name:road_loss(probes[name](capture.values[name],(512,512)),target) for name in names}
            sum(losses.values()).backward();optimizer.step()
            if any(parameter.grad is not None for parameter in model.parameters()):raise RuntimeError('B received gradients')
            if j%10==0 or j+1==len(fit):
                print(f'[PROBE FIT] epoch={epoch+1}/{a.probe_epochs} crop={j+1}/{len(fit)} loss={float(sum(losses.values()).detach()):.4f}',flush=True)
            loss_rows.append(dict(epoch=epoch+1,crop=j+1,**{n:float(v.detach()) for n,v in losses.items()}))
    csv_out(out/'probe_fit_losses.csv',loss_rows)
    torch.save(dict(probes=probes.state_dict(),metadata=stage_rows,args=vars(a),checkpoint_sha256=digest(a.checkpoint)),out/'readouts.pth')
    # Readout thresholds use whole training images held out from fitting.
    ts=np.round(np.arange(.05,.951,.01),2)
    totals={name:np.zeros((len(ts),3),dtype=np.int64) for name in names}
    base_counts=Counter()
    with torch.no_grad():
        for j,index in enumerate(held):
            sample=training[index];capture.clear();base=model(sample['image'][None].to(device))[0].sigmoid()[0,0].cpu().numpy()
            gt=sample['mask'][0].numpy()>.5;base_counts.update(count(base,gt,a.threshold))
            for name in names:
                prob=probes[name](capture.values[name],(512,512)).sigmoid()[0,0].cpu().numpy()
                for k,t in enumerate(ts):totals[name][k]+=list(count(prob,gt,t).values())
            print(f'[TRAIN HOLDOUT] {j+1}/{len(held)}',flush=True)
    thresholds={};curves=[];target_recall=metrics(base_counts)['recall']
    for name in names:
        rows=[dict(stage=name,threshold=float(t),**metrics(dict(zip(('tp','fp','fn'),map(int,c))))) for t,c in zip(ts,totals[name])]
        curves.extend(rows);best=max(rows,key=lambda r:(r['iou'],-r['threshold']))
        matched=min(rows,key=lambda r:(abs(r['recall']-target_recall),-r['threshold']))
        thresholds[name]=dict(own_holdout=best['threshold'],matched_holdout_recall=matched['threshold'])
    csv_out(out/'holdout_threshold_curves.csv',curves)
    dump(out/'thresholds.json',dict(selected_on='train_image_holdout',base_threshold=a.threshold,
        target_holdout_recall=target_recall,stages=thresholds))
    # Full native1500 val, same 512/256 weighted-logit blending as B evaluation.
    validation=RoadSkeletonDataset(str(a.root_path),'val',image_size=None,source_patch_size=1500,return_full_image=True)
    reference=json.loads(a.fixed_gaps.read_text());all_names=names+['actual_B']
    csv_rows=[];gap_rows=[];pictures=[];remaining=a.case_panels
    weight=blend_weight(512,device)[0,0].cpu().numpy()
    with torch.no_grad():
        for image_index in range(len(validation)):
            sample=validation[image_index];case=sample['case_name'];gt=sample['mask'][0].numpy()>.5
            if gt.shape!=(1500,1500) or case not in reference['images']:raise ValueError('Validation/fixed gap reference mismatch')
            canvases={name:np.zeros(gt.shape,np.float32) for name in all_names};divisor=np.zeros(gt.shape,np.float32)
            image=sample['image'][None].to(device)
            for top in sliding_positions(1500,512,256):
                for left in sliding_positions(1500,512,256):
                    capture.clear();base=model(image[:,:,top:top+512,left:left+512])[0]
                    sl=np.s_[top:top+512,left:left+512]
                    canvases['actual_B'][sl]+=base[0,0].cpu().numpy()*weight
                    for name in names:
                        z=probes[name](capture.values[name],(512,512))[0,0].cpu().numpy()
                        canvases[name][sl]+=z*weight
                    divisor[sl]+=weight
            probs={name:1/(1+np.exp(-np.clip(value/divisor,-80,80))) for name,value in canvases.items()}
            base=probs['actual_B']>=a.threshold
            skeleton=sample['skeleton'][0].numpy()>.5
            missed=skeleton&~(cv2.dilate(base.astype(np.uint8),np.ones((5,5),np.uint8))>0)
            road_dist=cv2.distanceTransform((~gt).astype(np.uint8),cv2.DIST_L2,5)
            near_background=(~gt)&(road_dist>3)&(road_dist<=64)
            for name in all_names:
                choices={'same_threshold':a.threshold}
                if name!='actual_B':choices.update(thresholds[name])
                for setting,t in choices.items():
                    pred=probs[name]>=t;c=count(probs[name],gt,t)
                    csv_rows.append(dict(case=case,stage=name,setting=setting,threshold=t,**c,**metrics(c),
                        missed_skeleton_n=int(missed.sum()),missed_skeleton_tp=int((pred&missed).sum()),
                        missed_probability_sum=float(probs[name][missed].sum(dtype=np.float64)),
                        near_background_n=int(near_background.sum()),near_background_fp=int((pred&near_background).sum())))
                for gap in reference['images'][case]['gaps']:
                    gap_rows.append(dict(case=case,gap_id=gap['id'],stage=name,threshold=a.threshold,
                        local_closed=local_closed(probs[name]>=a.threshold,gap),
                        skeleton_coverage=float((probs[name]>=a.threshold).reshape(-1)[gap['pixels']].mean())))
                write_image(out/'predictions'/name/(case+'.png'),np.round(probs[name]*255).astype(np.uint8))
            overview=cv2.cvtColor(rgb(sample),cv2.COLOR_RGB2BGR)
            for gap in reference['images'][case]['gaps']:
                yy,xx=np.divmod(np.array(gap['pixels']),1500);center=(int(xx.mean()),int(yy.mean()))
                cv2.circle(overview,center,12,(0,0,255),2)
                cv2.putText(overview,str(gap['id']),(center[0]+13,center[1]),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2)
            write_image(out/'overviews'/(case+'.jpg'),overview)
            per_image_limit=min(remaining,math.ceil(remaining/max(len(validation)-image_index,1)))
            picture_rows=gallery(out,case,rgb(sample),gt,reference['images'][case],['actual_B']+names,per_image_limit)
            pictures.extend(picture_rows);remaining-=len(picture_rows)
            print(f'[VAL STAGES] {image_index+1}/{len(validation)} {case}',flush=True)
    csv_out(out/'per_image.csv',csv_rows);csv_out(out/'gap_status.csv',gap_rows);csv_out(out/'case_panels.csv',pictures)
    summary=[]
    for name in all_names:
        for setting in ('same_threshold','own_holdout','matched_holdout_recall'):
            selected=[r for r in csv_rows if r['stage']==name and r['setting']==setting]
            if not selected:continue
            summed={key:sum(r[key] for r in selected) for key in ('tp','fp','fn','missed_skeleton_n','missed_skeleton_tp','missed_probability_sum','near_background_n','near_background_fp')}
            summary.append(dict(stage=name,setting=setting,threshold=selected[0]['threshold'],**metrics(summed),
                missed_skeleton_recall=summed['missed_skeleton_tp']/max(summed['missed_skeleton_n'],1),
                missed_skeleton_mean=summed['missed_probability_sum']/max(summed['missed_skeleton_n'],1),
                near_background_fpr=summed['near_background_fp']/max(summed['near_background_n'],1)))
    csv_out(out/'stage_metrics.csv',summary)
    lookup={(r['stage'],r['setting']):r for r in summary}
    transitions=[]
    for i in range(1,4):
        transitions.append(('downsample',f'enc{i}_blocks_before_down',f'enc{i}_after_down'))
        transitions.append(('encoder_blocks',f'enc{i}_after_down',f'enc{i+1}_blocks_before_down' if i<3 else 'enc4_after_blocks'))
    for i in (1,2):
        transitions.append(('upsample',f'dec{i}_blocks_before_up',f'dec{i}_after_up'))
        transitions.append(('decoder_blocks',f'dec{i}_input_after_skip',f'dec{i}_blocks_before_up'))
    transitions += [('first_upsample','dec0_input_no_skip','dec0_after_up'),
        ('skip_fusion','dec0_after_up','dec1_input_after_skip'),
        ('skip_fusion','dec1_after_up','dec2_input_after_skip'),
        ('skip_fusion','dec2_after_structure','dec3_input_after_skip'),
        ('DCA_skip_refinement','enc1_after_down','dec2_DCA_refined_skip'),
        ('DCA_skip_refinement','patch_embed','dec3_DCA_refined_skip'),
        ('global_H','H_stage2_before_global','H_stage2_after_global'),
        ('local_H','H_stage3_before_relation','H_final_after_relation'),
        ('decoder_normalization','dec3_after_structure','decoder_final_norm'),
        ('final_x4','decoder_final_norm','final_x4_after_up'),
        ('H_G3_surface_fusion','final_x4_after_up','final_H_G3_surface_fusion'),
        ('surface_refinement','final_H_G3_surface_fusion','surface_refine_before_boundary'),
        ('boundary_correction','surface_refine_before_boundary','surface_head_input_after_boundary')]
    drops=[]
    for kind,left,right in transitions:
        if left not in names or right not in names:continue
        first=lookup[left,'own_holdout'];second=lookup[right,'own_holdout']
        drops.append(dict(transition=kind,before=left,after=right,iou_delta=second['iou']-first['iou'],
            missed_recall_delta=second['missed_skeleton_recall']-first['missed_skeleton_recall'],
            near_background_fpr_delta=second['near_background_fpr']-first['near_background_fpr']))
    csv_out(out/'adjacent_stage_changes.csv',drops)
    capture.close()
    final_digest=model_digest(model)
    if final_digest!=original_digest:raise RuntimeError('Frozen B parameters or buffers changed')
    dump(out/'checks.json',dict(checkpoint_sha256=digest(a.checkpoint),model_before=original_digest,
        model_after=final_digest,model_unchanged=True,hook_output_max_abs=hook_delta,
        base_parameters_have_grad=any(p.grad is not None for p in model.parameters()),
        elapsed_seconds=time.perf_counter()-t0,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}))
    headers=list(summary[0]);table='<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in headers)+'</tr>'
    for row in summary:
        table+='<tr>'+''.join('<td>'+html.escape(f'{row[k]:.5f}' if isinstance(row[k],float) else str(row[k]))+'</td>' for k in headers)+'</tr>'
    table+='</table>'
    overview_links=' '.join(f'<a href="overviews/{html.escape(validation.image_files[i].rsplit(".",1)[0].replace("_sat",""))}.jpg">{html.escape(validation.image_files[i])}</a>' for i in range(len(validation)))
    links=''.join(f'<p>{html.escape(str(r["case"]))} gap {r["gap_id"]}</p><a href="{r["file"]}"><img loading="lazy" src="{r["file"]}" width="1100"></a>' for r in pictures)
    (out/'index.html').write_text('<meta charset="utf-8"><style>body{font:14px sans-serif}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:5px}img{max-width:100%;height:auto}</style><h1>原 B 逐层道路信息诊断</h1><p>全部彩色面板为独立线性读出概率，统一0至1：蓝色低、红色高；白线为GT道路边界。actual_B为原模型真实输出。不是把原始特征幅值当道路概率。</p><p>读出头只在训练图上拟合；阈值只在训练图像留出集确定。相邻表项包含分支切换，不能把整张表当单一串行网络。线性读出下降是定位线索，不证明信息不可恢复。</p><h2>原影像缺口标记</h2>'+overview_links+table+links,encoding='utf-8')
    (out/'REPORT.md').write_text('# 原 B 逐层道路信息诊断\n\n- B 权重与所有缓冲区未变化，hooks输出与原forward一致。\n- stages.csv记录实际执行顺序、通道数和分辨率；未截断特征通道。\n- stage_metrics.csv比较同阈值、训练留出集阈值、留出集匹配Recall。\n- gap_status.csv沿用原缺口ID/锚点，仅提供局部闭合代理；不等于人工确认补通。\n- 每层均采用相同1x1线性读出架构；输入通道和参数量不同，详见stages.csv。\n- 相邻行可能来自不同分支。重点对照enc*_blocks_before_down/after_down，以及dec*_blocks_before_up/after_up。\n- 各层低分既可能来自特征，也可能来自线性读出能力、训练不足与空间分辨率限制。不能直接宣称道路在该层不可逆丢失。\n- 本诊断不计算或冒充完整六项拓扑。\n',encoding='utf-8')
    print('[DONE]',out,flush=True)


if __name__=='__main__':main()
