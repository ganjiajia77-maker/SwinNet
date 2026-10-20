#!/usr/bin/env python3
"""Joint e202 surface-gap repair experiment for native 1500px Massachusetts."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, time, subprocess, sys, shutil, tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.spatial import cKDTree

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset, set_dataset_epoch
from losses.cldice_loss import SoftCLDiceLoss
from networks.vision_transformer import SwinUnet, STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626
from networks.surface_gap_repair import (GapCandidate, SurfaceGapRepair, candidate_tensors,
    generate_candidates, label_candidate, sliding_positions, stitch_logits, taper_weight, zhang_suen)


BASE_COMMIT="eb2da24d768b7f643e4fb288e3736693862c1fe8"
MODEL_NAME="e202_surface_gap_repair_mass1500_random512_fp32_joint80"
MEAN=np.array([0.485,0.456,0.406],np.float32)
STD=np.array([0.229,0.224,0.225],np.float32)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["train","resume","sweep","test","topology","baseline-only"])
    p.add_argument("--root_path",default="/home/gjj/Swin-Unet-main/Massachusetts/tiff")
    p.add_argument("--output_dir",default="/home/gjj/SwinNet-roadbias/model_out")
    p.add_argument("--run_name",default="")
    p.add_argument("--checkpoint",default="")
    p.add_argument("--pretrain_ckpt",default="/home/gjj/Swin-Unet-main/pretrained_ckpt/swinv2_tiny_patch4_window8_256.pth")
    p.add_argument("--cfg",default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    p.add_argument("--gpu",type=int,default=2)
    p.add_argument("--seed",type=int,default=1234)
    p.add_argument("--max_epochs",type=int,default=80)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--lambda_repair",type=float,default=1.0)
    p.add_argument("--lambda_cldice",type=float,default=0.1)
    p.add_argument("--accept_threshold",type=float,default=0.5)
    p.add_argument("--candidate_threshold",type=float,default=0.45)
    p.add_argument("--candidate_threshold_source",default="unverified historical source; configured at 0.45")
    p.add_argument("--baseline_only",action="store_true")
    p.add_argument("--candidate_replay_dir",default="",
        help="Replay train_candidates_epoch_*.json from a joint run (baseline-only control only)")
    p.add_argument("--limit_train_images",type=int,default=0,help="smoke-check only; 0 means all")
    return p

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def state_hash(module):
    digest=hashlib.sha256()
    for name,tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8")); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def save_checkpoint_atomic(checkpoint,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent)
    os.close(fd)
    try:
        torch.save(checkpoint,tmp)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def read_native(root,split,name):
    ds=RoadSkeletonDataset(root_dir=str(root),split=split,image_size=512,source_patch_size=1500,
        augment=False,return_full_image=True,surface_only=True)
    ip=Path(ds.image_dir)/name
    lp=Path(ds.mask_dir)/ds._find_label_name(name,ds.mask_dir)
    image=cv2.cvtColor(cv2.imread(str(ip),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    mask=cv2.imread(str(lp),cv2.IMREAD_GRAYSCALE)
    source_size=(1500,1500)
    image=ds._center_crop_or_pad(image,source_size)
    mask=ds._center_crop_or_pad(mask,source_size)
    rgb=(image.astype(np.float32)/255.0-MEAN)/STD
    rgb=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()
    mask=torch.from_numpy((mask>127).astype(np.float32))[None]
    return rgb,mask

def build_base(args,device,load_imagenet=True):
    cfg=get_config(argparse.Namespace(cfg=args.cfg,opts=None,batch_size=2,img_size=512,zip=False,
        cache_mode="",resume="",accumulation_steps=1,use_checkpoint=False,amp_opt_level="",tag="",
        eval=False,throughput=False))
    cfg.defrost(); cfg.DATA.IMG_SIZE=512; cfg.MODEL.PRETRAIN_CKPT=args.pretrain_ckpt; cfg.freeze()
    model=SwinUnet(cfg,num_classes=1,return_skeleton=True,
        structure_profile=STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
        enable_highres_structure_stream=True,highres_structure_channels=64,
        highres_structure_fuse_stages="stage23",highres_structure_fusion_mode="stage23",
        enable_e128_stage_fusion=True,enable_h3_surface_fusion=True,
        remove_stage2_pre_topology_source=True,enable_global_topology=True,
        global_topology_max_nodes=32,global_topology_heads=4,global_topology_alpha_max=0.05,
        stage_skeleton_mode="direct",surface_only=True).to(device)
    loaded=set()
    if load_imagenet:
        if not Path(args.pretrain_ckpt).is_file(): raise FileNotFoundError(args.pretrain_ckpt)
        loaded=model.load_from(cfg)
    return model,loaded

def base_forward(model,image,features=False):
    out=model(image,return_surface_features=features)
    feat=None
    if features: out,feat=out
    if isinstance(out,tuple): out=out[0]
    return out,feat

def tile_inference(model,rgb,device,need_features=False):
    h,w=rgb.shape[-2:]; pos=[(y,x) for y in sliding_positions(h,512,256) for x in sliding_positions(w,512,256)]
    logits=[]
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    if device.type=="cuda": torch.cuda.synchronize(device)
    started=time.perf_counter()
    for top,left in pos:
        tile=rgb[:,top:top+512,left:left+512][None].to(device)
        with torch.no_grad(): out,_=base_forward(model,tile,False)
        logits.append(out.detach())
    stitched=stitch_logits(logits,pos,(h,w))
    if device.type=="cuda": torch.cuda.synchronize(device)
    timing={"base_sliding_inference_seconds":time.perf_counter()-started,
            "peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else 0}
    return stitched,pos,logits,timing

def _read_case_ids(dataset,limit=0):
    names=list(dataset.image_files)
    return names[:limit] if limit else names

def _candidate_windows(rgb,mask,model,device,split,name,threshold=.45):
    base,pos,tile_logits,timing=tile_inference(model,rgb,device)
    prob=torch.sigmoid(base)[0,0].cpu().numpy()
    started=time.perf_counter()
    candidates,stats=generate_candidates(prob,threshold=threshold,tile=512,stride=256)
    labeled=[]
    gt=mask[0].numpy().astype(bool)
    gt_skeleton=zhang_suen(gt)
    ys,xs=np.nonzero(gt_skeleton)
    gt_tree=cKDTree(np.stack([ys,xs],axis=1)) if len(ys) else None
    gt_band=cv2.dilate(gt.astype(np.uint8),np.ones((9,9),np.uint8))>0
    base_corr=cv2.dilate((prob>=threshold).astype(np.uint8),np.ones((3,3),np.uint8))
    for cand in candidates:
        labeled.append(label_candidate(cand,gt,prob,gt_skeleton=gt_skeleton,gt_tree=gt_tree,
                                       gt_band=gt_band,base_corridor=base_corr))
    timing["candidate_generation_and_label_seconds"]=time.perf_counter()-started
    return base,pos,tile_logits,labeled,{**stats,**timing}

def _make_candidate_input(candidate,rgb,base_logit,features,device):
    top,left=candidate.tile_top,candidate.tile_left
    tile_rgb=rgb[:,top:top+512,left:left+512][None].to(device)
    tile_logit=base_logit
    if tile_logit.ndim==3: tile_logit=tile_logit[None]
    prompts,geom=candidate_tensors(candidate,device)
    return tile_rgb,torch.sigmoid(tile_logit),prompts,geom,features["encoder_stage1"],features["decoder_final"]

def _load_current_tile(model,rgb,candidate,device):
    top,left=candidate.tile_top,candidate.tile_left
    tile=rgb[:,top:top+512,left:left+512][None].to(device)
    out,features=base_forward(model,tile,True)
    return tile,out,features

def _window_map(size=512,device="cpu"):
    y=torch.arange(size,device=device,dtype=torch.float32)
    margin=torch.minimum(torch.minimum(y[:,None],(size-1-y)[:,None]),
                         torch.minimum(y[None,:],(size-1-y)[None,:]))
    return (margin/32.0).clamp(0,1)[None,None]

def repair_image(model,repair,rgb,device,candidate_threshold=.45,accept_threshold=.5,thresholds=(.45,),emit_dir=None):
    base,pos,tile_logits,timing=tile_inference(model,rgb,device)
    prob=torch.sigmoid(base)[0,0].cpu().numpy()
    started=time.perf_counter()
    candidates,stats=generate_candidates(prob,threshold=candidate_threshold,tile=512,stride=256)
    stats["candidate_generation_seconds"]=time.perf_counter()-started
    bytile=defaultdict(list)
    for c in candidates: bytile[(c.tile_top,c.tile_left)].append(c)
    repaired_tiles=[x.clone() for x in tile_logits]
    base_index={p:i for i,p in enumerate(pos)}
    cand_rows=[]
    if repair is not None:
        repair.eval()
        started=time.perf_counter()
        with torch.no_grad():
            for (top,left),items in bytile.items():
                tile,base_tile,features=_load_current_tile(model,rgb,items[0],device)
                deltas=[]; probs=[]; windows=[]
                for c in items:
                    inputs=_make_candidate_input(c,rgb,base_tile,features,device)
                    cls,delta=repair(*inputs)
                    p=torch.sigmoid(cls)
                    deltas.append(delta); probs.append(p); windows.append(_window_map(device=device))
                    cand_rows.append({**c.to_dict(),"accept_probability":float(p.cpu()),"accepted":bool(p>=accept_threshold)})
                numerator=torch.zeros_like(base_tile); denominator=torch.zeros_like(base_tile)
                for d,p,w,c in zip(deltas,probs,windows,items):
                    if float(p) < accept_threshold: continue
                    numerator += w*p.reshape(1,1,1,1)*d
                    denominator += w
                merged=base_tile+numerator/denominator.clamp_min(1.0)
                repaired_tiles[base_index[(top,left)]]=merged.detach()
        if device.type=="cuda": torch.cuda.synchronize(device)
        stats["feature_and_local_repair_seconds"]=time.perf_counter()-started
    else:
        stats["feature_and_local_repair_seconds"]=0.0
    repaired=stitch_logits(repaired_tiles,pos,rgb.shape[-2:])
    stats.update(timing)
    stats["candidate_count_after_tile_assignment"]=len(candidates)
    stats["peak_allocated_bytes"]=max(stats.get("peak_allocated_bytes",0),
        int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else 0)
    if emit_dir:
        emit_dir=Path(emit_dir); (emit_dir/"surface").mkdir(parents=True,exist_ok=True)
        torch.save({"base_logits":base.cpu(),"repaired_logits":repaired.cpu()},emit_dir/("logits.pt"))
    return base,repaired,stats,cand_rows

def focal_dice(logits,target,mask=None,gamma=1.0):
    target=target.float(); prob=torch.sigmoid(logits)
    bce=F.binary_cross_entropy_with_logits(logits,target,reduction="none")
    pt=prob*target+(1-prob)*(1-target)
    focal=((1-pt).pow(gamma)*bce)
    if mask is None: mask=torch.ones_like(target)
    denom=mask.sum().clamp_min(1)
    focal=(focal*mask).sum()/denom
    intersection=(prob*target*mask).sum((1,2,3))
    union=(prob*mask).sum((1,2,3))+(target*mask).sum((1,2,3))
    dice=1-((2*intersection+1.0)/(union+1.0)).mean()
    return focal+0.5*dice

def masked_soft_cldice(logits,target,mask,loss_fn=None):
    # Match losses.cldice_loss logits convention, but mask only after full-map skeletonization.
    from losses.cldice_loss import soft_skeletonize
    with torch.autocast(device_type=logits.device.type,enabled=False):
        p=torch.sigmoid(logits.float()); y=target.float()
        sp=soft_skeletonize(p,iter_num=10); sy=soft_skeletonize(y,iter_num=10)
        m=mask.float()
        tp=(sp*y*m).sum((1,2,3)); ps=(sp*m).sum((1,2,3))
        ts=(sy*p*m).sum((1,2,3)); gs=(sy*m).sum((1,2,3))
        precision=(tp+1e-6)/(ps+1e-6); recall=(ts+1e-6)/(gs+1e-6)
        cl=2*precision*recall/(precision+recall+1e-8)
        return (1-cl).mean()

def load_repair_checkpoint(args,device):
    ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    model,_=build_base(args,device,load_imagenet=False)
    model.load_state_dict(ckpt["model_state_dict"],strict=True)
    repair=None
    if ckpt.get("repair_state_dict") is not None:
        repair=SurfaceGapRepair().to(device)
        repair.load_state_dict(ckpt["repair_state_dict"],strict=True)
    return model.eval(),repair.eval(),ckpt

def save_mask(path,mask):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    cv2.imencode(path.suffix or ".png",(mask.astype(np.uint8)*255))[1].tofile(str(path))

def local_path_connected(mask,a,b,radius=3):
    """8-connected test inside the corridor ROI, with small anchor disks."""
    h,w=mask.shape; pad=8
    y0=max(0,min(a[0],b[0])-pad); y1=min(h,max(a[0],b[0])+pad+1)
    x0=max(0,min(a[1],b[1])-pad); x1=min(w,max(a[1],b[1])+pad+1)
    roi=mask[y0:y1,x0:x1].astype(np.uint8)
    aa=(a[0]-y0,a[1]-x0); bb=(b[0]-y0,b[1]-x0)
    mark_a=np.zeros_like(roi); mark_b=np.zeros_like(roi)
    cv2.circle(mark_a,(aa[1],aa[0]),radius,1,-1); cv2.circle(mark_b,(bb[1],bb[0]),radius,1,-1)
    labels_count,labels=cv2.connectedComponents(roi,connectivity=8)
    la=set(np.unique(labels[mark_a>0])); lb=set(np.unique(labels[mark_b>0])); la.discard(0); lb.discard(0)
    return bool(la & lb)

def evaluate_export(args,device,split,thresholds,selection_threshold=None):
    model,repair,ckpt=load_repair_checkpoint(args,device)
    dataset=RoadSkeletonDataset(args.root_path,split=split,image_size=512,source_patch_size=1500,
        augment=False,return_full_image=True,surface_only=True)
    outroot=Path(args.output_dir)/args.run_name/"evaluation"/split
    if outroot.exists() and any(outroot.glob("threshold_*/surface/*")):
        raise FileExistsError(f"Prediction masks already exist under {outroot}; use a new run or move the old evaluation directory")
    summary=[]
    for name in _read_case_ids(dataset):
        rgb,mask=read_native(Path(args.root_path),split,name)
        base,repaired,stats,cands=repair_image(model,repair,rgb,device,
            ckpt.get("protocol",{}).get("candidate_threshold",.45),
            ckpt.get("protocol",{}).get("accept_threshold",.5),thresholds)
        stem=Path(name).stem.replace("_sat","")
        base_prob=torch.sigmoid(base)[0,0].cpu().numpy()
        repaired_prob=torch.sigmoid(repaired)[0,0].cpu().numpy()
        truth=mask[0].numpy().astype(bool)
        base_binary=base_prob>=.45; repaired_binary=repaired_prob>=.45
        base_tp=int((base_binary&truth).sum()); base_fp=int((base_binary&~truth).sum())
        repair_tp=int((repaired_binary&truth).sum()); repair_fp=int((repaired_binary&~truth).sum())
        diagnostics={"candidate_count":len(cands),"candidate_count_pre_truncation":stats.get("proposed",0),
            "candidate_count_post_truncation":stats.get("kept",0),
            "new_tp":max(repair_tp-base_tp,0),"deleted_tp":max(base_tp-repair_tp,0),
            "new_fp":max(repair_fp-base_fp,0),"deleted_fp":max(base_fp-repair_fp,0),
            "accepted":0,"accepted_complete_local_path":0,"accepted_not_connected":0,
            "gt_supported_added_path":0,"gt_unsupported_added_path":0,"uncertain_added_path":0,
            "by_type":{},"by_gap_length":{}}
        gt_for_labels=truth
        gt_skeleton=zhang_suen(gt_for_labels); ys,xs=np.nonzero(gt_skeleton)
        gt_tree=cKDTree(np.stack([ys,xs],axis=1)) if len(ys) else None
        gt_band=cv2.dilate(gt_for_labels.astype(np.uint8),np.ones((9,9),np.uint8))>0
        base_corr=cv2.dilate((base_prob>=.45).astype(np.uint8),np.ones((3,3),np.uint8))
        for row in cands:
            d={k:row[k] for k in GapCandidate.__dataclass_fields__}
            d["anchor_a"]=tuple(d["anchor_a"]); d["anchor_b"]=tuple(d["anchor_b"])
            d["tangent_a"]=tuple(d["tangent_a"]); d["tangent_b"]=tuple(d["tangent_b"])
            c=GapCandidate(**d)
            lab=label_candidate(c,gt_for_labels,base_prob,gt_skeleton=gt_skeleton,
                                gt_tree=gt_tree,gt_band=gt_band,base_corridor=base_corr)
            row["offline_gt_label_for_diagnosis_only"]=lab.label
            row["offline_gt_corridor_coverage_for_diagnosis_only"]=lab.gt_coverage
            row["offline_base_local_gap_for_diagnosis_only"]=lab.predicted_gap
            row["gt_coverage"]=lab.gt_coverage
            row["predicted_gap"]=lab.predicted_gap
            bucket=f"{int(lab.distance//16)*16}-{int(lab.distance//16)*16+16}"
            diagnostics["by_type"].setdefault(c.kind,{"count":0,"accepted":0,"positive_labels":0,"negative_labels":0})
            diagnostics["by_type"][c.kind]["count"]+=1
            diagnostics["by_type"][c.kind]["positive_labels"]+=int(lab.label==1)
            diagnostics["by_type"][c.kind]["negative_labels"]+=int(lab.label==0)
            diagnostics["by_gap_length"].setdefault(bucket,{"count":0,"accepted":0,"positive_labels":0})
            diagnostics["by_gap_length"][bucket]["count"]+=1
            diagnostics["by_gap_length"][bucket]["positive_labels"]+=int(lab.label==1)
            if row.get("accepted"):
                diagnostics["accepted"]+=1
                diagnostics["by_type"][c.kind]["accepted"]+=1
                diagnostics["by_gap_length"][bucket]["accepted"]+=1
                pad=16; y0=max(0,min(c.anchor_a[0],c.anchor_b[0])-pad); y1=min(1500,max(c.anchor_a[0],c.anchor_b[0])+pad+1)
                x0=max(0,min(c.anchor_a[1],c.anchor_b[1])-pad); x1=min(1500,max(c.anchor_a[1],c.anchor_b[1])+pad+1)
                complete=local_path_connected(repaired_binary[y0:y1,x0:x1],(c.anchor_a[0]-y0,c.anchor_a[1]-x0),(c.anchor_b[0]-y0,c.anchor_b[1]-x0))
                if complete: diagnostics["accepted_complete_local_path"]+=1
                else: diagnostics["accepted_not_connected"]+=1
                if complete and lab.label==1: diagnostics["gt_supported_added_path"]+=1
                elif complete and lab.label==0: diagnostics["gt_unsupported_added_path"]+=1
                elif complete: diagnostics["uncertain_added_path"]+=1
        for t in thresholds:
            save_mask(outroot/f"threshold_{t:.2f}"/"surface"/f"{stem}.png",
                      (torch.sigmoid(repaired)[0,0].cpu().numpy()>=t))
        basep=torch.sigmoid(base)[0,0].cpu().numpy()
        save_mask(outroot/"base_threshold_0.45"/"surface"/f"{stem}.png",basep>=.45)
        arrays=outroot/"raw_logits"; arrays.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(arrays/f"{stem}.npz",base_logits=base[0,0].cpu().numpy().astype(np.float32),
                            repaired_logits=repaired[0,0].cpu().numpy().astype(np.float32))
        vis=np.zeros((1500,1500,3),np.uint8); vis[base_binary]=(70,70,220); vis[repaired_binary]=(70,210,70)
        vis[truth]=(220,70,70); vis[(repaired_binary&~base_binary)]=(0,255,255)
        visdir=outroot/"visualizations"; visdir.mkdir(parents=True,exist_ok=True)
        cv2.imencode(".png",vis)[1].tofile(str(visdir/f"{stem}.png"))
        (outroot/"candidates").mkdir(parents=True,exist_ok=True)
        (outroot/"candidates"/f"{stem}.json").write_text(json.dumps({"generation":stats,"candidates":cands,"diagnostics":diagnostics},indent=2),encoding="utf-8")
        summary.append({"case":stem,"candidate_stats":stats,"candidates":len(cands),"diagnostics":diagnostics,
            "timings_seconds":{"base_sliding_inference":stats.get("base_sliding_inference_seconds",0),
              "candidate_generation":stats.get("candidate_generation_seconds",0),
              "features_and_local_repair":stats.get("feature_and_local_repair_seconds",0)},
            "peak_allocated_bytes":stats.get("peak_allocated_bytes",0)})
        print(f"[{split}] {stem}: {len(cands)} candidates",flush=True)
    mean=lambda key: float(np.mean([row["timings_seconds"][key] for row in summary])) if summary else 0.0
    aggregate={"metric_tool":"road-mask-v1.1","checkpoint_epoch":ckpt.get("epoch"),
        "repair_parameters":sum(p.numel() for p in repair.parameters()) if repair else 0,
        "images":len(summary),"mean_candidates_per_image":float(np.mean([x["candidates"] for x in summary])) if summary else 0,
        "mean_base_sliding_inference_seconds":mean("base_sliding_inference"),
        "mean_candidate_generation_seconds":mean("candidate_generation"),
        "mean_feature_and_local_repair_seconds":mean("features_and_local_repair"),
        "max_peak_allocated_bytes":max([x["peak_allocated_bytes"] for x in summary],default=0),
        "device":torch.cuda.get_device_name(device) if device.type=="cuda" else "cpu",
        "per_image":summary}
    (outroot/"inference_summary.json").write_text(json.dumps(aggregate,indent=2),encoding="utf-8")
    return outroot

def train(args,device,resume=False,baseline_only=False):
    seed_all(args.seed)
    run=args.run_name or f"e202_surface_gap_repair_mass1500_random512_fp32_joint80_s{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir=Path(args.output_dir)/run; run_dir.mkdir(parents=True,exist_ok=True)
    free_bytes=shutil.disk_usage(run_dir).free
    if free_bytes < 5*(1024**3):
        raise RuntimeError(f"Need at least 5 GiB free for safe last/best checkpoint writes; available={free_bytes/1024**3:.2f} GiB")
    args.run_name=run
    # Recreate the exact ImageNet/new optimizer grouping before restoring a run.
    model,loaded=build_base(args,device,load_imagenet=True)
    repair=None if baseline_only else SurfaceGapRepair().to(device)
    optim_params=[]; loaded_params=[]
    loaded=set(loaded)
    for name,p in model.named_parameters():
        if p.requires_grad: (loaded_params if name in loaded else optim_params).append(p)
    if repair is not None: optim_params+=list(repair.parameters())
    optimizer=torch.optim.AdamW([{"params":loaded_params,"lr":5e-5,"initial_lr":5e-5,"name":"imagenet_loaded"},
        {"params":optim_params,"lr":2e-4,"initial_lr":2e-4,"name":"new_and_repair"}],weight_decay=.01)
    start=0; best=-1
    if resume:
        path=args.checkpoint or str(run_dir/"last.pth")
        ckpt=torch.load(path,map_location="cpu",weights_only=False)
        if ckpt.get("base_commit")!=BASE_COMMIT: raise ValueError("Resume checkpoint has wrong base commit")
        model.load_state_dict(ckpt["model_state_dict"],strict=True)
        if repair is not None: repair.load_state_dict(ckpt["repair_state_dict"],strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"]); start=int(ckpt["epoch"]); best=float(ckpt.get("best_val_iou",-1))
        loaded=set(ckpt["loaded_pretrained_names"])
    train_ds=RoadSkeletonDataset(args.root_path,split="train",image_size=512,source_patch_size=1500,
        random_crop_train=True,random_crops_per_image=1,random_crop_seed=args.seed,augment=True,surface_only=True)
    names=_read_case_ids(train_ds,args.limit_train_images)
    train_ds.image_files=names
    loader=DataLoader(train_ds,batch_size=2,shuffle=True,num_workers=args.workers,
        pin_memory=True,persistent_workers=args.workers>0)
    pre_sha=hashlib.sha256(Path(args.pretrain_ckpt).read_bytes()).hexdigest()
    loaded_names=sorted(loaded)
    (run_dir/"provenance.json").write_text(json.dumps({"model":MODEL_NAME,"base_commit":BASE_COMMIT,
        "pretrain_path":args.pretrain_ckpt,"pretrain_sha256":pre_sha,"loaded_pretrained_names":loaded_names,
        "candidate_protocol":{"threshold":args.candidate_threshold,"threshold_source":args.candidate_threshold_source,
          "8_connectivity":8,"tangent_native_px":16,
          "distance_native_px":[8,128],"angle_max_degrees":75,"boundary_margin_px":8,
          "max_per_endpoint":8,"max_candidates_image":256,"prompt_sigma_native_px":3},
        "training":{"epochs":args.max_epochs,"batch_size":2,"accumulation":1,"seed":args.seed,
          "precision":"FP32","ema":False,"train_mode":"joint","ordinary_crop_per_image_epoch":1,
          "candidate_tiles_per_step":2,"lambda_repair":args.lambda_repair,"lambda_cldice":args.lambda_cldice,
          "softcldice":{"iterations":10,"smooth":1e-6,"mask_safe_border":32}},
        "repair_parameters":sum(p.numel() for p in repair.parameters()) if repair else 0},indent=2),encoding="utf-8")
    cldice_fn=SoftCLDiceLoss(iter_num=10,smooth=1e-6)
    for epoch in range(start,args.max_epochs):
        for group in optimizer.param_groups:
            initial=group["initial_lr"]; minimum=5e-6 if group["name"]=="imagenet_loaded" else 1e-5
            if epoch<10: lr=initial*(epoch+1)/10
            else:
                progress=min((epoch-10)/max(args.max_epochs-10-1,1),1.)
                lr=minimum+(initial-minimum)*.5*(1+math.cos(math.pi*progress))
            group["lr"]=lr
        model.train()
        if repair: repair.train()
        set_dataset_epoch(train_ds,epoch)
        # Refresh candidates once per epoch using current base weights, eval/no-grad only.
        candidate_groups=defaultdict(list); candidate_counts=[]; candidate_records=[]
        if baseline_only and args.candidate_replay_dir:
            replay_path=Path(args.candidate_replay_dir)/f"train_candidates_epoch_{epoch+1:03d}.json"
            if not replay_path.is_file():
                raise FileNotFoundError(f"Missing joint-run train candidate manifest: {replay_path}")
            replay=json.loads(replay_path.read_text(encoding="utf-8"))
            if replay.get("split")!="train" or int(replay.get("epoch",-1))!=epoch+1:
                raise ValueError(f"Candidate replay has mismatched split/epoch: {replay_path}")
            train_name_set=set(names)
            for row in replay.get("candidates",[]):
                image_name=row["image"]
                if image_name not in train_name_set:
                    raise ValueError(f"Replay manifest contains non-train image {image_name!r}")
                fields={k:row[k] for k in GapCandidate.__dataclass_fields__}
                for key in ("anchor_a","anchor_b","tangent_a","tangent_b"):
                    fields[key]=tuple(fields[key])
                c=GapCandidate(**fields); candidate_records.append({"image":image_name,**c.to_dict()})
                candidate_groups[(image_name,c.tile_top,c.tile_left)].append(c)
            # Match the joint run's tile exposure: a tile is sampled only when
            # at least one candidate in that tile has a non-ignored label.
            candidate_groups={key:items for key,items in candidate_groups.items()
                              if any(c.label>=0 for c in items)}
            candidate_manifest={"epoch":epoch+1,"split":"train","source":"replayed_joint_train_manifest",
                "source_manifest":str(replay_path.resolve()),"source_weights_sha256":replay.get("weights_sha256"),
                "weights_sha256":state_hash(model),"exposed_tiles":len(candidate_groups),
                "candidates":candidate_records}
        else:
            prior=model.training; model.eval()
            with torch.no_grad():
                for image_name in names:
                    rgb,gt=read_native(Path(args.root_path),"train",image_name)
                    _,_,_,items,stats=_candidate_windows(rgb,gt,model,device,"train",image_name,args.candidate_threshold)
                    for c in items:
                        candidate_records.append({"image":image_name,**c.to_dict()})
                    valid_tiles={(image_name,c.tile_top,c.tile_left) for c in items if c.label>=0}
                    for c in items:
                        key=(image_name,c.tile_top,c.tile_left)
                        # Keep every generated proposal in any tile that has a supervised candidate.
                        if key in valid_tiles: candidate_groups[key].append(c)
                    candidate_counts.append({"image":image_name,**stats,"labeled":sum(c.label>=0 for c in items),
                        "positive":sum(c.label==1 for c in items),"negative":sum(c.label==0 for c in items),
                        "ignored":sum(c.label<0 for c in items)})
            model.train(prior)
            candidate_manifest={"epoch":epoch+1,"split":"train","weights_sha256":state_hash(model),
              "protocol":{"threshold":args.candidate_threshold,"max_per_endpoint":8,"max_candidates_image":256,
                "distance_px":[8,128],"tangent_px":16,"boundary_px":8},
              "images":candidate_counts,"exposed_tiles":len(candidate_groups),"candidates":candidate_records}
        (run_dir/f"train_candidates_epoch_{epoch+1:03d}.json").write_text(json.dumps(candidate_manifest,indent=2),encoding="utf-8")
        candidate_pool=list(candidate_groups.items())
        random.Random(args.seed+epoch).shuffle(candidate_pool); candidate_cursor=0
        epoch_loss=0.; nstep=0; cldice_skips=0; seen_candidates=0
        for batch in loader:
            images=batch["image"].to(device,non_blocking=True); targets=batch["mask"].to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits,_=base_forward(model,images,False)
            base_loss=focal_dice(logits,targets)
            repair_loss=logits.sum()*0.0
            # Candidate patches are additional, normalized inputs; each unique tile is used once per step.
            selected=[]
            target_tiles=min(2,len(candidate_pool))
            while candidate_pool and len(selected)<target_tiles:
                if candidate_cursor>=len(candidate_pool): candidate_cursor=0; random.shuffle(candidate_pool)
                key,items=candidate_pool[candidate_cursor]; candidate_cursor+=1
                if key not in [k for k,_ in selected]: selected.append((key,items))
            if selected:
                conn_terms=[]; pixel_terms=[]; cl_terms=[]
                selected_labels=[c.label for _,items in selected for c in items if c.label>=0]
                npos=sum(x==1 for x in selected_labels); nneg=sum(x==0 for x in selected_labels)
                for (image_name,top,left),items in selected:
                    rgb,gt=read_native(Path(args.root_path),"train",image_name)
                    tile,base_tile,features=_load_current_tile(model,rgb,items[0],device)
                    gt_tile=gt[:,top:top+512,left:left+512][None].to(device)
                    if repair is not None:
                        numerator=torch.zeros_like(base_tile); denominator=torch.zeros_like(base_tile)
                        region=torch.zeros_like(base_tile); local_conn=[]
                        for c in items:
                            inp=_make_candidate_input(c,rgb,base_tile,features,device)
                            cls,delta=repair(*inp); p=torch.sigmoid(cls); win=_window_map(device=device)
                            numerator += win*p.reshape(1,1,1,1)*delta
                            denominator += win
                            region=torch.maximum(region,(win>0).float())
                            if c.label>=0:
                                class_weight=(0.5/npos if c.label==1 else 0.5/nneg) if npos and nneg else 1.0/max(len(selected_labels),1)
                                local_conn.append(F.binary_cross_entropy_with_logits(cls,torch.tensor([float(c.label)],device=device),reduction="none").mean()*class_weight)
                        seen_candidates+=len(items)
                        repaired=base_tile+numerator/denominator.clamp_min(1.0)
                        pix=focal_dice(repaired,gt_tile,region)
                        safe=region.clone(); safe[:,:,:32,:]=0; safe[:,:,-32:,:]=0; safe[:,:,:,:32]=0; safe[:,:,:,-32:]=0
                        if safe.any(): cl=masked_soft_cldice(repaired,gt_tile,safe,cldice_fn)
                        else: cl=repaired.sum()*0; cldice_skips+=1
                        conn_terms.extend(local_conn); pixel_terms.append(pix); cl_terms.append(cl)
                    else:
                        region=torch.zeros_like(base_tile)
                        for c in items:
                            region=torch.maximum(region,(_window_map(device=device)>0).float())
                        safe=region.clone(); safe[:,:,:32,:]=0; safe[:,:,-32:,:]=0; safe[:,:,:,:32]=0; safe[:,:,:,-32:]=0
                        pixel_terms.append(focal_dice(base_tile,gt_tile,region)); cl_terms.append(masked_soft_cldice(base_tile,gt_tile,safe,cldice_fn))
                        seen_candidates+=len(items)
                conn_loss=torch.stack(conn_terms).sum() if conn_terms else logits.sum()*0
                pixel_loss=torch.stack(pixel_terms).mean() if pixel_terms else logits.sum()*0
                cl_loss=torch.stack(cl_terms).mean() if cl_terms else logits.sum()*0
                repair_loss=conn_loss+args.lambda_repair*(pixel_loss+args.lambda_cldice*cl_loss)
            loss=base_loss+repair_loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters())+(list(repair.parameters()) if repair else []),1.0)
            # Epoch-wise warmup and cosine, independently for ImageNet-loaded and new groups.
            optimizer.step(); epoch_loss+=float(loss.detach()); nstep+=1
            if nstep%10==0: print(f"epoch={epoch+1} step={nstep}/{len(loader)} loss={float(loss.detach()):.4f} base={float(base_loss.detach()):.4f} repair={float(repair_loss.detach()):.4f}",flush=True)
        val_iou=validate_small(model,repair,args,device,limit=0)
        ckpt={"base_commit":BASE_COMMIT,"epoch":epoch+1,"best_val_iou":max(best,val_iou),
            "model_state_dict":model.state_dict(),"repair_state_dict":repair.state_dict() if repair else None,
            "optimizer_state_dict":optimizer.state_dict(),"loaded_pretrained_names":loaded_names,
            "pretrain_ckpt":args.pretrain_ckpt,"pretrain_sha256":pre_sha,
            "repair_parameter_count":sum(p.numel() for p in repair.parameters()) if repair else 0,
            "epoch_train_steps":nstep,"candidate_tiles_seen":seen_candidates,
            "candidate_mining_weights_sha256":candidate_manifest["weights_sha256"],
            "scheduler_state_dict":{"name":"warmup_cosine","epoch_completed":epoch+1,
                "max_epochs":args.max_epochs,"warmup_epochs":10,"pretrained_lr":5e-5,
                "pretrained_min_lr":5e-6,"new_lr":2e-4,"new_min_lr":1e-5},
            "repair_weight_sha256":state_hash(repair) if repair else None,
            "protocol":{"candidate_threshold":args.candidate_threshold,"candidate_threshold_source":args.candidate_threshold_source,
                "accept_threshold":args.accept_threshold,
                "lambda_repair":args.lambda_repair,"lambda_cldice":args.lambda_cldice,
                "proposal":{"connectivity":8,"skeleton":"Zhang-Suen","tangent_length_px":16,
                    "min_distance_px":8,"max_distance_px":128,"angle_max_degrees":75,
                    "boundary_margin_px":8,"max_per_endpoint":8,"max_candidates_image":256,
                    "tile_size":512,"tile_stride":256,"anchor_context_px":16,"prompt_sigma_px":3},
                "candidate_refresh":"once at each epoch start, current eval/no_grad weights",
                "candidate_cache_file":f"train_candidates_epoch_{epoch+1:03d}.json",
                "split":"train","features":["encoder_stage1_before_patch_merging","decoder_final_before_surface_head"],
                "tile_merge":"sum(W*p*delta)/max(1,sum(W)); taper0.1 weighted logit stitch",
                "soft_cldice":{"formula":"1-harmonic(precision,recall)","iterations":10,
                    "smooth":1e-6,"safe_border_px":32,"input":"logits","dtype":"fp32"}},
            "args":vars(args),"val_iou":val_iou}
        save_checkpoint_atomic(ckpt,run_dir/"last.pth")
        if val_iou>best:
            best=val_iou; ckpt["best_val_iou"]=best; save_checkpoint_atomic(ckpt,run_dir/"best.pth")
        print(f"[epoch {epoch+1}] loss={epoch_loss/max(nstep,1):.5f} candidates_seen={seen_candidates} cldice_skips={cldice_skips} val_iou={val_iou:.5f} best={best:.5f}",flush=True)
    print(f"BEST={run_dir/'best.pth'}",flush=True)

def validate_small(model,repair,args,device,limit=0):
    # Global IoU at the fixed base threshold, used only to select a checkpoint.
    ds=RoadSkeletonDataset(args.root_path,split="val",image_size=512,source_patch_size=1500,
        augment=False,return_full_image=True,surface_only=True)
    names=_read_case_ids(ds,limit); tp=fp=fn=0
    model.eval(); repair.eval() if repair else None
    with torch.no_grad():
        for name in names:
            rgb,gt=read_native(Path(args.root_path),"val",name)
            _,logits,_,_=repair_image(model,repair,rgb,device)
            pred=torch.sigmoid(logits)>=.45; truth=gt[None].to(device)>0.5
            tp+=int((pred&truth).sum()); fp+=int((pred&~truth).sum()); fn+=int((~pred&truth).sum())
    return tp/max(tp+fp+fn,1)

def write_manifest(args,checkpoint,split,thresholds,output):
    base=Path(__file__).resolve().parent
    predroot=Path(args.output_dir)/args.run_name/"evaluation"/split
    predictions=[{"threshold":float(t),"path":str(predroot/f"threshold_{t:.2f}"/"surface")} for t in thresholds]
    ckpt=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model_name=MODEL_NAME if ckpt.get("repair_state_dict") is not None else MODEL_NAME+"_baseline_only"
    provenance={"checkpoint":str(Path(checkpoint).resolve()),"base_commit":BASE_COMMIT,
        "epoch":ckpt.get("epoch"),"candidate_threshold":args.candidate_threshold,
        "candidate_threshold_source":args.candidate_threshold_source,
        "accept_threshold":ckpt.get("protocol",{}).get("accept_threshold",.5),
        "inference":"native1500;512/256 tiles; taper0.1 weighted logits; single candidate pass",
        "prediction_generation":"all candidates from unrepaired P0; tile repair merged by weighted mean",
        "repair_parameters":sum(x.numel() for x in SurfaceGapRepair().parameters()) if ckpt.get("repair_state_dict") is not None else 0,
        "metric_tool":"road-mask-v1.1 raster APLS approximation"}
    models=[{"name":model_name,"provenance":provenance,"predictions":{split:predictions}}]
    if split=="val" and ckpt.get("repair_state_dict") is not None:
        baseline_masks=[{"threshold":.45,"path":str(predroot/"base_threshold_0.45"/"surface")}]
        models.append({"name":MODEL_NAME+"_base_same_checkpoint","provenance":{**provenance,
            "prediction_generation":"repair module bypassed; same trained joint checkpoint"},
            "predictions":{"val":baseline_masks}})
        models.append({"name":MODEL_NAME+"_fixed045","provenance":{**provenance,
            "prediction_generation":"repaired logits at fixed 0.45 for equal-threshold comparison"},
            "predictions":{"val":[{"threshold":.45,"path":str(predroot/"threshold_0.45"/"surface")}]}})
    elif split=="test" and ckpt.get("repair_state_dict") is not None:
        baseline_masks=[{"threshold":.45,"path":str(predroot/"base_threshold_0.45"/"surface")}]
        models.append({"name":MODEL_NAME+"_base_same_checkpoint","provenance":{**provenance,
            "prediction_generation":"repair module bypassed; same trained joint checkpoint"},
            "predictions":{"test":baseline_masks}})
        models.append({"name":MODEL_NAME+"_fixed045","provenance":{**provenance,
            "prediction_generation":"repaired logits at fixed 0.45 for equal-threshold comparison"},
            "predictions":{"test":[{"threshold":.45,"path":str(predroot/"threshold_0.45"/"surface")}]}})
    manifest={"data_root":str(Path(args.root_path).resolve()),
      "protocol":{"image_size":1500,"short_area_threshold":20,"apls_max_nodes":64,"apls_snap_radius":5.0},
      "models":models}
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(manifest,indent=2),encoding="utf-8")

def run_unified(args,split,thresholds,selection=None):
    run_dir=Path(args.output_dir)/args.run_name
    predroot=run_dir/"evaluation"/split
    manifest=run_dir/f"manifest_{split}.json"
    write_manifest(args,args.checkpoint,split,thresholds,manifest)
    report=predroot/"unified_report"
    cmd=[sys.executable,str(Path(__file__).parent/"evaluation/road_comparison/compare_predictions.py"),
         "--manifest",str(manifest),"--split",split,"--output_dir",str(report),"--progress_every","1"]
    if split=="val": cmd.append("--select_threshold_only")
    else: cmd.extend(["--selection",str(selection)])
    subprocess.run(cmd,cwd=Path(__file__).parent,check=True)

def main():
    args=parser().parse_args()
    if args.mode=="baseline-only": args.baseline_only=True; args.mode="train"
    device=torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.mode in ("train","resume"):
        train(args,device,resume=args.mode=="resume",baseline_only=args.baseline_only); return
    if not args.checkpoint: raise SystemExit("--checkpoint is required for evaluation modes")
    if not args.run_name: args.run_name=Path(args.checkpoint).parent.name
    if args.mode=="sweep":
        thresholds=[round(x*.05,2) for x in range(1,20)]
        root=evaluate_export(args,device,"val",thresholds)
        args.checkpoint=str(Path(args.checkpoint).resolve())
        run_unified(args,"val",thresholds)
        print(f"VAL_MASKS={root}")
        return
    if args.mode=="test":
        sel=Path(args.output_dir)/args.run_name/"evaluation"/"val"/"unified_report"/"val_selection.json"
        if not sel.is_file(): raise SystemExit("Run unified validation selection first")
        ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        selected_name=MODEL_NAME if ckpt.get("repair_state_dict") is not None else MODEL_NAME+"_baseline_only"
        threshold=json.loads(sel.read_text())["models"][selected_name]["threshold"]
        root=evaluate_export(args,device,"test",sorted(set([threshold,.45])),threshold)
        run_unified(args,"test",[threshold],sel)
        print(f"TEST_MASKS={root}"); return
    if args.mode=="topology":
        report=Path(args.output_dir)/args.run_name/"evaluation"/"test"/"unified_report"
        csv_path=report/"test_comparison.csv"
        if not csv_path.is_file(): raise SystemExit("Run test first; unified topology report is missing")
        print(f"Topology report already computed by road-mask-v1.1: {csv_path}")

if __name__=="__main__": main()
