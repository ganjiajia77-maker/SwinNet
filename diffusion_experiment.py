"""Massachusetts H0 Stage2 skeleton / Stage3 diffusion train and unified evaluation."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from diffusion_protocol import (ARCHITECTURE_ID, BASE_COMMIT, ROOT, architecture, build_model,
                                initialize_from_h0, load_checkpoint, save_atomic, write_json,
                                H256_ID, EXPERIMENT_BASE)
from losses.h0_diffusion_loss import H0DiffusionLoss
from mass_protocol import stitch_logits
from networks.directional_diffusion import MODES


def fixed_args(args):
    return dict(vars(args), img_size=512, source_patch_size=1500, random_crop_train=True,
        random_crops_per_image=1, direct_resize_train=False, overlap_stride=256,
        batch_size=2, accumulation_steps=1, max_epochs=80, use_ema=False, amp_dtype='none',
        warmup_epochs=10, pretrained_lr=5e-5, pretrained_min_lr=5e-6, new_lr=2e-4, new_min_lr=1e-5,
        augment=True, val_interval=5, threshold=.5, trend_val_every_steps=0,
        structure_profile='stage23_boundary_0626', stage_skeleton_mode='direct',
        stage2_skeleton_gradient_ratio=.5, stage3_skeleton_gradient_ratio=.5,
        effective_aux_weights={'S2': .008, 'S3': .012, 'D3': .0024},
        stage2_skeleton_weight=.008, stage3_skeleton_weight=.012, stage_direction_factor=.2,
        stage_skeleton_loss_factor=1., stage_skeleton_only_loss_factor=1.,
        surface_focal_gamma=1., surface_dice_weight=.5, skeleton_pos_weight=2.,
        highres_structure_skeleton_weight=0., road_attention_weight=0., skeleton_stage_weight=0.,
        final_skeleton_weight=0., final_connectivity_weight=0., boundary_aux_weight=0.,
        enable_global_topology=False, remove_stage2_pre_topology_source=True,
        enable_highres_structure_stream=True, highres_structure_channels=64,
        enable_h3_surface_fusion=True, baseline_args_source='published fe3b730 command; not a server checkpoint audit')


def learning_rate(epoch, peak, floor):
    if epoch < 10:
        return peak*(epoch+1)/10
    progress = min((epoch-10)/69, 1.)
    return floor+(peak-floor)*.5*(1.+math.cos(math.pi*progress))


def dataset(args, split, training=False):
    return RoadSkeletonDataset(args.root_path, split=split, source_patch_size=1500,
        image_size=512 if training else None, return_full_image=not training,
        augment=training, random_crop_train=training, random_crops_per_image=1,
        random_crop_seed=args.seed, tile_size=512 if training else None, tile_stride=256)


def seed_worker(worker):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def benchmark(model, device):
    model.eval()
    image = torch.zeros(1, 3, 512, 512, device=device)
    with torch.inference_mode():
        model(image)
        if device.type == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        model(image)
        if device.type == 'cuda': torch.cuda.synchronize()
    return {'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'inference_ms_one_pass': (time.perf_counter()-start)*1000,
        'benchmark_input': [1,3,512,512], 'precision': 'FP32', 'warmup_passes': 1,
        'device': torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU',
        'torch_version': torch.__version__, 'timing_scope': 'model only; no I/O; no claims about full1500 timing'}


@torch.inference_mode()
def validate(model, loader, device, criterion):
    model.eval()
    tp = fp = fn = 0
    losses = []
    for batch in loader:
        image, target = batch['image'].to(device), batch['mask'].to(device)
        logits, _ = stitch_logits(model, image)
        loss, _ = criterion(logits, surface_gt=target)
        losses.append(float(loss))
        pred, gt = logits.sigmoid() >= .5, target > .5
        tp += int((pred & gt).sum())
        fp += int((pred & ~gt).sum())
        fn += int((~pred & gt).sum())
    return {'val_loss_surface_only': float(np.mean(losses)), 'iou': tp/(tp+fp+fn+1e-8),
            'f1': 2*tp/(2*tp+fp+fn+1e-8), 'precision': tp/(tp+fp+1e-8), 'recall': tp/(tp+fn+1e-8)}


def train(args, arc, device):
    run_dir = Path(args.output_dir)/args.run_name
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f'Use a fresh run directory: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    if args.resume:
        model, restored = load_checkpoint(args.resume)
        if restored['architecture_config'] != arc or restored['args']['seed'] != args.seed:
            raise ValueError('Resume mode/settings/seed must equal saved configuration')
        loaded = set(restored['loaded_pretrained_names'])
        migration = restored['initialization_manifest']
        start_epoch, best_f1 = restored['epoch'], restored['best_f1']
        generator.set_state(restored['loader_generator_state'])
    else:
        model = build_model(arc)
        loaded, migration = initialize_from_h0(model, arc, args.seed, args.pretrain_ckpt,
                                              args.h0_init_ckpt, args.base_init_ckpt)
        start_epoch, best_f1 = 0, -1.
    write_json(run_dir/'initialization_manifest.json', migration)
    parsed = fixed_args(args)
    write_json(run_dir/'resolved_config.json', dict(args=parsed, architecture=arc, base_h0_commit=BASE_COMMIT,
        base_experiment_commit=EXPERIMENT_BASE if arc['architecture_id'] == H256_ID else None,
        implementation=implementation_revision(), model_config=model.config.dump()))
    model = model.to(device)
    inspection = benchmark(model, device)
    write_json(run_dir/'pretrain_inspection.json', inspection)
    print(json.dumps(inspection, ensure_ascii=False), flush=True)
    print(f'Architecture={arc["architecture_id"]}; mode={args.diffusion_mode}; C2/D2/C3=disabled/null; '
          'old gates=disabled; Stage2 H->decoder=disabled; global nodes=disabled; '
          'loss=surface+0.008*S2+0.012*S3+0.0024*D3; EMA=off; batch2/acc1', flush=True)
    groups = []
    for name, peak, floor, select in (
        ('imagenet_loaded', 5e-5, 5e-6, True), ('random_or_custom', 2e-4, 1e-5, False)):
        params = [p for key,p in model.named_parameters() if p.requires_grad and ((key in loaded) == select)]
        if params: groups.append(dict(params=params, group_name=name, lr=peak, peak=peak, floor=floor))
    optimizer = torch.optim.AdamW(groups, weight_decay=.0001)
    if args.resume: optimizer.load_state_dict(restored['optimizer_state_dict'])
    training, validation = dataset(args, 'train', True), dataset(args, 'val')
    worker_options = dict(persistent_workers=True, prefetch_factor=2) if args.num_workers else {}
    train_loader = DataLoader(training, batch_size=2, shuffle=True, generator=generator,
        num_workers=args.num_workers, worker_init_fn=seed_worker, pin_memory=device.type=='cuda', **worker_options)
    val_loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=args.num_workers,
        worker_init_fn=seed_worker, pin_memory=device.type=='cuda', **worker_options)
    criterion = H0DiffusionLoss().to(device)
    log_path = run_dir/'losses.csv'
    columns = ['epoch','lr','loss','surface_loss','S2','S3','D3','C2','D2','C3','val_iou','val_f1']
    with log_path.open('a' if args.resume else 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not args.resume: writer.writeheader()
        for epoch in range(start_epoch, 80):
            training.set_epoch(epoch)
            model.train()
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(epoch, group['peak'], group['floor'])
            sums = {key:0. for key in ('loss','surface_loss','S2','S3','D3')}
            count = 0
            for step,batch in enumerate(train_loader):
                image, target, skeleton = (batch[key].to(device) for key in ('image','mask','skeleton'))
                optimizer.zero_grad(set_to_none=True)
                outputs = model(image)
                loss, terms = criterion(outputs[0], surface_gt=target, skeleton_gt=skeleton, stage_outputs=outputs[4])
                if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite batch loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not bool(torch.isfinite(norm)): raise FloatingPointError('Nonfinite gradients')
                optimizer.step()
                sums['loss'] += float(loss.detach())
                for key in ('surface_loss','S2','S3','D3'): sums[key] += float(terms[key])
                count += 1
                if step % 10 == 0:
                    print(f'Epoch {epoch+1}/80 batch {step+1}/{len(train_loader)} loss={float(loss.detach()):.5f}', flush=True)
            metrics = validate(model, val_loader, device, criterion) if (epoch+1)%5==0 or epoch==79 else None
            means = {key:value/max(count,1) for key,value in sums.items()}
            improved = metrics is not None and metrics['f1'] > best_f1
            if improved: best_f1 = metrics['f1']
            writer.writerow(dict(epoch=epoch+1, lr=optimizer.param_groups[-1]['lr'], **means,
                C2='disabled',D2='disabled',C3='disabled',val_iou=metrics['iou'] if metrics else None,
                val_f1=metrics['f1'] if metrics else None))
            handle.flush()
            checkpoint = dict(architecture_id=arc['architecture_id'], architecture_config=arc, base_h0_commit=BASE_COMMIT,
                base_experiment_commit=EXPERIMENT_BASE if arc['architecture_id'] == H256_ID else None,
                implementation=implementation_revision(),
                model_config=model.config.dump(),
                model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                args=parsed, epoch=epoch+1, best_f1=best_f1, val_metrics=metrics,
                initialization_manifest=migration, loaded_pretrained_names=sorted(loaded),
                loader_generator_state=generator.get_state())
            save_atomic(checkpoint, run_dir/'last.pth')
            if improved: save_atomic(checkpoint, run_dir/'best.pth')
            print(f'Epoch {epoch+1}: {means}; val={metrics}; deleted heads=disabled', flush=True)


def model_name(checkpoint):
    architecture_id = checkpoint.get('architecture_id', checkpoint['architecture_config']['architecture_id'])
    prefix = 'h0_encoder_h256_s2skel_s3diff_' if architecture_id == H256_ID else 'h0_s2skel_s3diff_'
    return prefix+checkpoint['architecture_config']['diffusion_mode']


def implementation_revision():
    marker = ROOT/'SOURCE_COMMIT'
    if marker.is_file():
        return marker.read_text().strip()
    try:
        top = subprocess.check_output(['git','rev-parse','--show-toplevel'], cwd=ROOT, text=True).strip()
        if Path(top).resolve() == ROOT.resolve():
            revision = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
            dirty = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'], cwd=ROOT, text=True).strip()
            return revision+('+uncommitted' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        pass
    return 'unavailable; see source_sha256'


@torch.inference_mode()
def export_masks(model, args, split, thresholds, device):
    import cv2
    model.eval()
    data = dataset(args, split)
    directories = {t:Path(args.output_dir)/'predictions'/split/f'threshold_{t:.2f}'/'surface' for t in thresholds}
    for path in directories.values():
        if path.exists() and any(path.iterdir()): raise FileExistsError(path)
        path.mkdir(parents=True, exist_ok=True)
    for index in range(len(data)):
        sample = data[index]
        logits,_ = stitch_logits(model, sample['image'][None].to(device))
        probability = logits.sigmoid()[0,0].cpu().numpy()
        for threshold,path in directories.items():
            destination = path/(sample['case_name']+'_pred.png')
            if not cv2.imwrite(str(destination), (probability>=threshold).astype(np.uint8)*255):
                raise OSError(destination)
        print(f'{split} {index+1}/{len(data)}: {sample["case_name"]}', flush=True)


def manifest(args, checkpoint, threshold=None):
    root = Path(args.output_dir)
    sources = lambda split: [{'threshold':float(p.parent.name.split('_',1)[1]), 'path':str(p.resolve())}
                             for p in sorted((root/'predictions'/split).glob('threshold_*/surface'))]
    provenance = dict(checkpoint=str(Path(args.model_path).resolve()), epoch=checkpoint['epoch'],
        architecture=checkpoint['architecture_config'], seed=checkpoint['args']['seed'], base_h0_commit=BASE_COMMIT,
        initialization=checkpoint['initialization_manifest']['shared_tensor_sha256'], weights='raw; no EMA',
        inference='512 windows / stride256; weighted logits floor0.1; native1500; no TTA/postprocessing',
        implementation=checkpoint.get('implementation'),base_experiment_commit=checkpoint.get('base_experiment_commit'))
    provenance['source_sha256'] = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
        'diffusion_experiment.py','diffusion_protocol.py','networks/directional_diffusion.py','networks/encoder_h256.py',
        'networks/skeleton_guided_head.py','networks/swin_transformer_unet_skip_expand_decoder_sys.py',
        'networks/vision_transformer.py','losses/h0_diffusion_loss.py','losses/road_losses.py',
        'datasets/dataset_road_skeleton.py','mass_protocol.py',
        'evaluation/road_comparison/road_metrics.py','evaluation/road_comparison/compare_predictions.py')}
    data = dict(data_root=str(Path(args.root_path).resolve()),
        protocol=dict(image_size=1500,short_area_threshold=20,apls_max_nodes=64,apls_snap_radius=5.),
        models=[dict(name=model_name(checkpoint),provenance=provenance,
                     predictions=dict(val=sources('val'), test=sources('test') if threshold is not None else []))])
    write_json(root/'comparison.json', data)
    return root/'comparison.json'


def compare(args, checkpoint, split, selection=None):
    path = manifest(args, checkpoint, 0. if split=='test' else None)
    output = Path(args.output_dir)/split
    command = [sys.executable,'-u',str(ROOT/'evaluation/road_comparison/compare_predictions.py'),
               '--manifest',str(path),'--split',split,'--output_dir',str(output),'--progress_every','10']
    if split=='val': command += ['--select_threshold_only']
    if selection: command += ['--selection',str(selection)]
    subprocess.run(command, check=True)


def main(architecture_id=ARCHITECTURE_ID):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['inspect','train','sweep','test','topology'])
    p.add_argument('--root_path', default='/home/gjj/Swin-Unet-main/Massachusetts/tiff')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--run_name', default='')
    p.add_argument('--pretrain_ckpt')
    p.add_argument('--h0_init_ckpt', help='explicit original H0 weight migration; never resume optimizer')
    p.add_argument('--base_init_ckpt', help='explicit a668711 weight migration for H256; no optimizer migration')
    p.add_argument('--model_path')
    p.add_argument('--resume')
    p.add_argument('--gpu', default='2')
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--num_workers',type=int,default=4)
    p.add_argument('--diffusion_mode',choices=MODES,default='semantic_directional')
    p.add_argument('--diffusion_steps',type=int,default=8)
    p.add_argument('--diffusion_eta',type=float,default=.1)
    p.add_argument('--diffusion_beta',type=float,default=.5)
    p.add_argument('--diffusion_latent_channels',type=int,default=32)
    p.add_argument('--encoder_h_exchange', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--enable_h_surface_residual', action=argparse.BooleanOptionalAction, default=True)
    args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    os.environ['OPENCV_LOG_LEVEL']='ERROR'
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device=torch.device('cpu' if args.gpu=='cpu' else 'cuda')
    arc=architecture(args.diffusion_mode,args.diffusion_steps,args.diffusion_eta,args.diffusion_beta,
                     args.diffusion_latent_channels,architecture_id,args.encoder_h_exchange,args.enable_h_surface_residual)
    if args.action in ('train','inspect'):
        if not args.pretrain_ckpt or not Path(args.pretrain_ckpt).is_file(): p.error('Provide original SwinV2 pretraining')
        if args.action=='inspect':
            model=build_model(arc)
            _,migration=initialize_from_h0(model,arc,args.seed,args.pretrain_ckpt,args.h0_init_ckpt,args.base_init_ckpt)
            output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
            report=benchmark(model.to(device),device)
            write_json(output/'inspection.json',report); write_json(output/'initialization_manifest.json',migration)
            print(json.dumps(report,ensure_ascii=False,indent=2))
        else:
            if args.resume:
                resume_dir=Path(args.resume).resolve().parent
                args.output_dir=str(resume_dir.parent); args.run_name=resume_dir.name
            prefix = 'mass_h256_s2skel_s3diff' if architecture_id == H256_ID else 'mass_s2skel_s3diff'
            args.run_name=args.run_name or f'{prefix}_{args.diffusion_mode}_s{args.seed}_'+datetime.now().strftime('%Y%m%d_%H%M%S')
            train(args,arc,device)
        return
    if not args.model_path: p.error('Provide a new architecture checkpoint')
    model,checkpoint=load_checkpoint(args.model_path)
    if checkpoint['architecture_id'] != architecture_id:
        raise ValueError('Use the entry matching this checkpoint architecture; no cross-architecture resume/inference')
    model=model.to(device).eval()
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    if args.action=='sweep':
        export_masks(model,args,'val',[round(i/100,2) for i in range(5,100,5)],device)
        compare(args,checkpoint,'val')
        print((output/'val/val_threshold_scores.csv').read_text())
        return
    selection=output/'val/val_selection.json'
    chosen=json.loads(selection.read_text())['models'][model_name(checkpoint)]['threshold']
    if args.action=='test':
        export_masks(model,args,'test',[chosen],device)
        compare(args,checkpoint,'test',selection)
    elif not (output/'test/test_comparison.csv').is_file():
        compare(args,checkpoint,'test',selection)
    print((output/'test/test_comparison.csv').read_text())


if __name__=='__main__':
    main()
