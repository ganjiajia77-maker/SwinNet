"""Frozen ten-probe experiment for the a4001b3 E2-direction H0 checkpoint.

The H0 checkpoint is never optimized. One backbone forward supplies ten
independent 1x1 road heads. Early stopping uses fixed validation crops;
threshold selection and final evaluation use native 1500/512/256 stitching.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Load the exact a4001b3 source tree before resolving config/networks. This
# script lives in the separate local-gap diagnostic checkout.
def _probe_repo_from_argv():
    for index, item in enumerate(sys.argv[1:], 1):
        if item == '--probe_repo' and index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1]).resolve()
        if item.startswith('--probe_repo='):
            return Path(item.split('=', 1)[1]).resolve()
    raise ValueError('--probe_repo must point to the exact a4001b3 code checkout')


SOURCE_ROOT = _probe_repo_from_argv()
if not (SOURCE_ROOT / 'h0_e2_config.py').is_file():
    raise FileNotFoundError(f'a4001b3 source not found: {SOURCE_ROOT}')
sys.path.insert(0, str(SOURCE_ROOT))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from h0_encoder_protocol import BASE_COMMIT, MODEL_NAME, blend_weight, sliding_positions
from h0_e2_config import validate_checkpoint
from networks.vision_transformer import SwinUnet


ROOT = SOURCE_ROOT
CFG = ROOT / 'configs' / 'swin_tiny_patch4_window7_224_lite.yaml'
PROTOCOL_VERSION = 'h0-a400-ten-road-probes-v1'
PROBES = ('E1', 'E2', 'E3', 'H0', 'H_E2', 'H_E3', 'F1', 'F2', 'F3', 'surface_prehead')
ALL_NAMES = (*PROBES, 'original_surface')
SEVEN_PROBES = ('E2', 'E3', 'H_E2', 'H_E3', 'F2', 'F3', 'surface_prehead')
SIZE, NATIVE, STRIDE = 512, 1500, 256
THRESHOLDS = tuple(round(i / 100, 2) for i in range(5, 100, 5))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def atomic_save(value, path):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def config_args():
    return SimpleNamespace(cfg=str(CFG), opts=None, batch_size=2, img_size=SIZE,
                           zip=False, cache_mode='', resume='', accumulation_steps=0,
                           use_checkpoint=False, amp_opt_level='', tag='', eval=True,
                           throughput=False)


def load_backbone(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected a4001b3 checkpoint dictionary')
    saved = validate_checkpoint(checkpoint)
    if saved.get('img_size') != SIZE:
        raise ValueError('Checkpoint must use 512 inputs')
    model = SwinUnet(
        config=get_config(config_args()), img_size=SIZE, num_classes=1,
        return_skeleton=True,
        bottleneck_type=saved.get('bottleneck_type', 'global_local'),
        final_topology_eta_init=saved.get('final_topology_eta_init', 0.005),
        final_gap_rho_init=saved.get('final_gap_rho_init', 0.005),
        structure_profile=saved['structure_profile'],
        stage2_skeleton_gradient_ratio=saved['stage2_skeleton_gradient_ratio'],
        stage3_skeleton_gradient_ratio=saved['stage3_skeleton_gradient_ratio'],
        stage3_gate_topology_gradient_ratio=saved['stage3_gate_topology_gradient_ratio'],
        final_skeleton_gradient_ratio=saved.get('final_skeleton_gradient_ratio', 0.0),
        enable_highres_structure_stream=saved['enable_highres_structure_stream'],
        highres_structure_channels=saved['highres_structure_channels'],
        highres_structure_fuse_stages=saved['highres_structure_fuse_stages'],
        highres_structure_fusion_mode=saved['highres_structure_fusion_mode'],
        enable_post_refine_structure_interaction=saved.get('enable_post_refine_structure_interaction', False),
        enable_global_topology=saved['enable_global_topology'],
        global_topology_max_nodes=saved['global_topology_max_nodes'],
        global_topology_heads=saved['global_topology_heads'],
        global_topology_alpha_max=saved['global_topology_alpha_max'],
        stage_skeleton_mode=saved['stage_skeleton_mode'],
        enable_h3_surface_fusion=saved['enable_h3_surface_fusion'],
        stage_skeleton_bias_init=saved.get('stage_skeleton_bias_init', 'zero'),
        stage_skeleton_positive_prior=saved.get('stage_skeleton_positive_prior', 0.05),
        remove_stage2_pre_topology_source=saved['remove_stage2_pre_topology_source'],
        encoder_structure_exchange=True,
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError('Backbone is not frozen')
    return model


class FeatureTap:
    def __init__(self, model):
        self.model = model
        self.values = {}
        backbone = model.swin_unet
        self.handles = [
            backbone.layers_up[1].register_forward_hook(self._f1_hook),
            backbone.decoder_structure_blocks['2'].register_forward_hook(self._decoder_hook('F2')),
            backbone.decoder_structure_blocks['3'].register_forward_hook(self._decoder_hook('F3')),
            backbone.guided_head.surface_head.register_forward_pre_hook(self._surface_hook),
        ]

    def _f1_hook(self, _module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise RuntimeError('F1 decoder output must be B,L,C tokens')
        batch, count, channels = output.shape
        side = int(count ** .5)
        if side * side != count:
            raise RuntimeError(f'F1 decoder tokens are not a square grid: {count}')
        self.values['F1'] = output.detach().transpose(1, 2).reshape(batch, channels, side, side)

    def _decoder_hook(self, name):
        def hook(_module, _inputs, outputs):
            self.values[name] = outputs[0].detach()
        return hook

    def _surface_hook(self, _module, inputs):
        # The global-topology prior also calls surface_head. Its later call in
        # guided_head overwrites this entry with the actual final-head input.
        self.values['surface_prehead'] = inputs[0].detach()

    def __call__(self, images):
        self.values.clear()
        with torch.no_grad():
            outputs = self.model(images)
        backbone = self.model.swin_unet
        for name in ('E1', 'E2', 'E3', 'H0', 'H_E2', 'H_E3'):
            self.values[name] = backbone.last_encoder_features[name].detach()
        if set(self.values) != set(PROBES):
            raise RuntimeError(f'Incomplete feature taps: {sorted(self.values)}')
        if any(value.ndim != 4 or value.shape[0] != images.shape[0]
               for value in self.values.values()):
            raise RuntimeError('Feature tap returned an invalid tensor')
        expected = {'E1': (96, 128, 128), 'E2': (192, 64, 64),
                    'H0': (64, 128, 128), 'H_E2': (64, 128, 128),
                    'H_E3': (64, 128, 128), 'F1': (192, 64, 64)}
        for name, shape in expected.items():
            if tuple(self.values[name].shape[1:]) != shape:
                raise RuntimeError(f'{name} shape {tuple(self.values[name].shape)} != {shape}')
        return dict(self.values), outputs[0].detach()

    def close(self):
        for handle in self.handles:
            handle.remove()


class RoadOnlyCrop(Dataset):
    """Same normalized 1500->random512 crops, without costly GT skeletonization."""

    def __init__(self, root, split, seed, max_train_images):
        self.source = RoadSkeletonDataset(root_dir=str(root), split=split,
                                          image_size=SIZE, source_patch_size=NATIVE)
        self.train = split == 'train'
        self.seed = seed
        count = len(self.source.image_files)
        if self.train and 0 < max_train_images < count:
            self.indices = np.linspace(0, count - 1, max_train_images, dtype=np.int64).tolist()
        else:
            self.indices = list(range(count))
        self.epoch_shared = torch.zeros(1, dtype=torch.int64).share_memory_()

    def set_epoch(self, epoch):
        self.epoch_shared[0] = int(epoch)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        name = self.source.image_files[source_index]
        label_name = self.source._find_label_name(name, self.source.mask_dir)
        image = cv2.imread(os.path.join(self.source.image_dir, name), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(self.source.mask_dir, label_name), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f'Could not read {name} or {label_name}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.source._center_crop_or_pad(image, (NATIVE, NATIVE))
        mask = self.source._center_crop_or_pad(mask, (NATIVE, NATIVE))
        if self.train:
            rng = np.random.RandomState((self.seed + int(self.epoch_shared[0]) * 1000003
                                         + source_index * 9176) % (2 ** 32))
            image, mask = self.source._augment_geometry(image, mask, rng=rng)
            image = self.source._augment_color_degrade(image, rng=rng)
            top, left = int(rng.randint(0, NATIVE - SIZE + 1)), int(rng.randint(0, NATIVE - SIZE + 1))
        else:
            top = left = (NATIVE - SIZE) // 2
        image = image[top:top + SIZE, left:left + SIZE]
        mask = mask[top:top + SIZE, left:left + SIZE]
        return normalize(image), torch.from_numpy((mask > 127).astype(np.float32)[None])


def normalize(rgb):
    image = rgb.astype(np.float32) / 255.0
    image = (image - np.array((0.485, 0.456, 0.406), dtype=np.float32)) / np.array(
        (0.229, 0.224, 0.225), dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def make_heads(channels, device):
    heads = nn.ModuleDict({name: nn.Conv2d(channels[name], 1, 1) for name in PROBES}).to(device)
    for head in heads.values():
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    return heads


def probe_logits(head, feature):
    logits = head(feature)
    if logits.shape[-2:] != (SIZE, SIZE):
        logits = F.interpolate(logits, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
    return logits


def road_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    focal = (bce * (1.0 - torch.exp(-bce))).mean()
    prob = torch.sigmoid(logits)
    dice = 1.0 - (2.0 * (prob * target).sum(dim=(1, 2, 3)) + 1.0) / (
        prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0)
    return focal + 0.5 * dice.mean()


def scores(tp, fp, fn):
    return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
            'iou': float(tp / (tp + fp + fn)) if tp + fp + fn else 0.0,
            'f1': float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            'precision': float(tp / (tp + fp)) if tp + fp else 0.0,
            'recall': float(tp / (tp + fn)) if tp + fn else 0.0}


def validate_crops(tap, heads, loader, device):
    totals = {name: np.zeros(3, dtype=np.int64) for name in PROBES}
    loss_totals = {name: 0.0 for name in PROBES}
    images_seen = 0
    with torch.no_grad():
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            images_seen += images.shape[0]
            features, _ = tap(images)
            for name in PROBES:
                logits = probe_logits(heads[name], features[name])
                loss_totals[name] += float(road_loss(logits, target)) * images.shape[0]
                pred = torch.sigmoid(logits) >= 0.5
                actual = target >= 0.5
                totals[name] += np.array([(pred & actual).sum().item(),
                                          (pred & ~actual).sum().item(),
                                          (~pred & actual).sum().item()], dtype=np.int64)
    return {name: {**scores(*totals[name]), 'val_loss': loss_totals[name] / images_seen}
            for name in PROBES}


def train(args, model, tap, backbone_hash, device):
    resuming = args.mode == 'resume'
    if resuming:
        last_path = args.run_dir / 'probe_last.pth'
        if not last_path.is_file():
            raise FileNotFoundError(f'No probe training state: {last_path}')
        last = torch.load(last_path, map_location='cpu', weights_only=False)
        if (last.get('protocol_version') != PROTOCOL_VERSION
                or last['backbone_sha256'] != backbone_hash or last['seed'] != args.seed
                or last['max_train_images'] != args.max_train_images
                or last['lr'] != args.lr or last['batch_size'] != args.batch_size):
            raise ValueError('Resume settings differ from the saved probe training state')
        start_epoch = int(last['epoch']) + 1
        if start_epoch > args.max_epochs:
            raise ValueError(f'Increase --max_epochs above {last["epoch"]} to resume')
    else:
        if args.run_dir.exists() and any(args.run_dir.iterdir()):
            raise FileExistsError(f'Training run directory is not empty: {args.run_dir}')
        args.run_dir.mkdir(parents=True, exist_ok=True)
        start_epoch = 1
    train_data = RoadOnlyCrop(args.root_path, 'train', args.seed, args.max_train_images)
    val_data = RoadOnlyCrop(args.root_path, 'val', args.seed, 0)
    worker_count = args.num_workers
    train_generator = torch.Generator()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=worker_count, pin_memory=True,
                              persistent_workers=worker_count > 0,
                              generator=train_generator)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=worker_count, pin_memory=True,
                            persistent_workers=worker_count > 0)
    first_image, _ = train_data[0]
    features, _ = tap(first_image.unsqueeze(0).to(device))
    channels = {name: features[name].shape[1] for name in PROBES}
    shapes = {name: list(features[name].shape[1:]) for name in PROBES}
    print(f'Frozen backbone features: {shapes}', flush=True)
    heads = make_heads(channels, device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=1e-4)
    backbone_ids = {id(parameter) for parameter in model.parameters()}
    if backbone_ids & {id(parameter) for group in optimizer.param_groups for parameter in group['params']}:
        raise RuntimeError('Backbone parameter entered probe optimizer')
    if resuming:
        if last['channels'] != channels:
            raise ValueError('Backbone feature channels changed since probe training')
        heads.load_state_dict(last['current_states'], strict=True)
        optimizer.load_state_dict(last['optimizer_state'])
        best_iou, best_epoch, best_states = last['best_iou'], last['best_epoch'], last['best_states']
        best_head_loss = last['best_head_loss']
        best_mean_loss = float(last['best_mean_loss'])
        with (args.run_dir / 'training_history.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        stale = int(last['stale'])
    else:
        best_iou = {name: -1.0 for name in PROBES}
        best_head_loss = {name: float('inf') for name in PROBES}
        best_mean_loss = float('inf')
        best_epoch = {}
        best_states = {}
        rows = []
        stale = 0
    for epoch in range(start_epoch, args.max_epochs + 1):
        train_data.set_epoch(epoch - 1)
        train_generator.manual_seed(args.seed + epoch)
        heads.train()
        total_loss = 0.0
        batches = 0
        for images, target in train_loader:
            images, target = images.to(device), target.to(device)
            features, _ = tap(images)
            optimizer.zero_grad(set_to_none=True)
            losses = [road_loss(probe_logits(heads[name], features[name]), target)
                      for name in PROBES]
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        heads.eval()
        validation = validate_crops(tap, heads, val_loader, device)
        mean_val_loss = float(np.mean([validation[name]['val_loss'] for name in PROBES]))
        improved = mean_val_loss < best_mean_loss - 1e-5
        if improved:
            best_mean_loss = mean_val_loss
        for name in PROBES:
            metric = validation[name]
            rows.append({'epoch': epoch, 'probe': name, 'train_loss_mean': total_loss / batches,
                         **metric})
            if (metric['iou'] > best_iou[name] + 1e-6
                    or (abs(metric['iou'] - best_iou[name]) <= 1e-6
                        and metric['val_loss'] < best_head_loss[name] - 1e-5)):
                best_iou[name] = metric['iou']
                best_head_loss[name] = metric['val_loss']
                best_epoch[name] = epoch
                best_states[name] = {key: value.detach().cpu().clone()
                                     for key, value in heads[name].state_dict().items()}
                improved = True
        stale = 0 if improved else stale + 1
        payload = {'schema': 1, 'protocol_version': PROTOCOL_VERSION,
                   'backbone_path': str(args.model_path),
                   'backbone_sha256': backbone_hash, 'base_commit': BASE_COMMIT,
                   'architecture_id': MODEL_NAME, 'channels': channels, 'feature_shapes': shapes,
                   'probe_names': PROBES, 'seed': args.seed,
                   'train_image_count': len(train_data), 'val_image_count': len(val_data),
                   'training_crop': 'one random512 crop per selected native1500 image per epoch',
                   'early_stop_validation': 'one fixed center512 crop per validation image; global IoU at 0.5',
                   'best_iou': best_iou, 'best_epoch': best_epoch, 'best_states': best_states}
        payload['best_head_loss'] = best_head_loss
        payload['best_mean_loss'] = best_mean_loss
        atomic_save(payload, args.run_dir / 'probe_best.pth')
        atomic_save({**payload, 'epoch': epoch, 'stale': stale,
                     'current_states': heads.state_dict(), 'optimizer_state': optimizer.state_dict(),
                     'max_train_images': args.max_train_images, 'lr': args.lr,
                     'batch_size': args.batch_size}, args.run_dir / 'probe_last.pth')
        write_csv(args.run_dir / 'training_history.csv', rows)
        print(f'Epoch {epoch}: train_loss={total_loss / batches:.5f}, '
              f'mean_val_iou={np.mean([validation[n]["iou"] for n in PROBES]):.5f}, '
              f'best_epochs={best_epoch}', flush=True)
        if epoch >= 5 and stale >= args.patience:
            print(f'Early stop after {stale} epochs without any head improving', flush=True)
            break
    print(f'Saved ten a400 probe heads: {args.run_dir / "probe_best.pth"}', flush=True)


def load_heads(path, backbone_hash, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload.get('protocol_version') != PROTOCOL_VERSION
            or payload.get('backbone_sha256') != backbone_hash
            or payload.get('architecture_id') != MODEL_NAME):
        raise ValueError('Probe heads were trained on a different H0 checkpoint')
    if tuple(payload.get('probe_names', ())) != PROBES:
        raise ValueError('Probe names differ from this code')
    heads = make_heads(payload['channels'], device)
    for name in PROBES:
        heads[name].load_state_dict(payload['best_states'][name], strict=True)
    return heads.eval(), payload


def native_cases(root, split):
    source = RoadSkeletonDataset(root_dir=str(root), split=split,
                                 image_size=SIZE, source_patch_size=NATIVE)
    for name in source.image_files:
        label_name = source._find_label_name(name, source.mask_dir)
        rgb = cv2.imread(os.path.join(source.image_dir, name), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(source.mask_dir, label_name), cv2.IMREAD_GRAYSCALE)
        if rgb is None or mask is None:
            raise FileNotFoundError(name)
        rgb = source._center_crop_or_pad(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), (NATIVE, NATIVE))
        mask = source._center_crop_or_pad(mask, (NATIVE, NATIVE))
        if rgb.shape[:2] != (NATIVE, NATIVE) or mask.shape != (NATIVE, NATIVE):
            raise ValueError(f'Invalid native image or GT size: {name}')
        yield Path(name).stem, rgb, mask > 127


@torch.no_grad()
def native_probabilities(tap, heads, rgb, device):
    weight = blend_weight(SIZE, device)
    denominator = torch.zeros((1, 1, NATIVE, NATIVE), device=device)
    canvases = {name: torch.zeros_like(denominator) for name in ALL_NAMES}
    image = normalize(rgb).unsqueeze(0).to(device)
    for top in sliding_positions(NATIVE, SIZE, STRIDE):
        for left in sliding_positions(NATIVE, SIZE, STRIDE):
            features, original = tap(image[:, :, top:top + SIZE, left:left + SIZE])
            predictions = {name: probe_logits(heads[name], features[name]) for name in PROBES}
            predictions['original_surface'] = original
            region = (slice(None), slice(None), slice(top, top + SIZE), slice(left, left + SIZE))
            denominator[region] += weight
            for name, logits in predictions.items():
                if logits.shape != (1, 1, SIZE, SIZE):
                    raise ValueError(f'Invalid {name} tile shape: {tuple(logits.shape)}')
                canvases[name][region] += logits.float() * weight
    if bool((denominator <= 0).any()):
        raise RuntimeError('Native stitching left uncovered pixels')
    return {name: torch.sigmoid(canvas / denominator)[0, 0].cpu().numpy()
            for name, canvas in canvases.items()}


def threshold_counts(prob, target):
    # Match NumPy's float32 array >= Python-float threshold semantics exactly.
    bins = np.searchsorted(np.asarray(THRESHOLDS, dtype=prob.dtype), prob, side='right')
    positive = np.bincount(bins[target].ravel(), minlength=len(THRESHOLDS) + 1)
    negative = np.bincount(bins[~target].ravel(), minlength=len(THRESHOLDS) + 1)
    tp = np.cumsum(positive[::-1])[::-1][1:]
    fp = np.cumsum(negative[::-1])[::-1][1:]
    fn = int(target.sum()) - tp
    return np.stack((tp, fp, fn), axis=1)


def sweep(args, tap, heads, backbone_hash, device):
    totals = {name: np.zeros((len(THRESHOLDS), 3), dtype=np.int64) for name in ALL_NAMES}
    count = 0
    for case, rgb, target in native_cases(args.root_path, 'val'):
        probabilities = native_probabilities(tap, heads, rgb, device)
        for name in ALL_NAMES:
            totals[name] += threshold_counts(probabilities[name], target)
        count += 1
        print(f'Validation {count}: {case}', flush=True)
    rows, chosen = [], {}
    for name in ALL_NAMES:
        candidates = []
        for threshold, (tp, fp, fn) in zip(THRESHOLDS, totals[name]):
            record = {'name': name, 'threshold': threshold, **scores(tp, fp, fn)}
            rows.append(record)
            candidates.append(record)
        best = max(candidates, key=lambda record: (record['iou'], -record['threshold']))
        chosen[name] = {'threshold': best['threshold'], 'val_iou': best['iou'],
                        'val_f1': best['f1']}
    write_csv(args.run_dir / 'val_threshold_scores.csv', rows)
    write_json(args.run_dir / 'val_selection.json', {
        'schema': 1, 'protocol_version': PROTOCOL_VERSION,
        'selected_on': 'val', 'criterion': 'global_iou',
        'backbone_sha256': backbone_hash, 'probe_checkpoint': str(args.run_dir / 'probe_best.pth'),
        'val_images': count, 'tile_size': SIZE, 'stride': STRIDE,
        'merge': 'weighted logits then sigmoid', 'thresholds': THRESHOLDS, 'models': chosen})
    print(json.dumps(chosen, indent=2), flush=True)


def test(args, tap, heads, probe_payload, backbone_hash, device):
    selected_path = args.run_dir / 'val_selection.json'
    if not selected_path.is_file():
        raise FileNotFoundError('Run sweep first: val_selection.json is missing')
    selection = json.loads(selected_path.read_text(encoding='utf-8'))
    if (selection.get('protocol_version') != PROTOCOL_VERSION
            or selection.get('selected_on') != 'val'
            or selection.get('backbone_sha256') != backbone_hash):
        raise ValueError('Validation selection belongs to another checkpoint')
    if set(selection['models']) != set(ALL_NAMES):
        raise ValueError('Validation selection is incomplete')
    output = args.run_dir / 'test_predictions'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Test predictions already exist: {output}')
    for name in ALL_NAMES:
        (output / name / 'surface').mkdir(parents=True, exist_ok=True)
    totals = {name: np.zeros(3, dtype=np.int64) for name in ALL_NAMES}
    skeleton_hits = {name: 0 for name in ALL_NAMES}
    baseline_missing_hits = {name: 0 for name in ALL_NAMES}
    baseline_missing_total = 0
    skeleton_total = 0
    count = 0
    from evaluation.road_comparison.road_metrics import skeletonize
    for case, rgb, target in native_cases(args.root_path, 'test'):
        probabilities = native_probabilities(tap, heads, rgb, device)
        skeleton = skeletonize(target)
        skeleton_total += int(skeleton.sum())
        baseline_threshold = selection['models']['original_surface']['threshold']
        baseline_missing = skeleton & (probabilities['original_surface'] < baseline_threshold)
        baseline_missing_total += int(baseline_missing.sum())
        for name in ALL_NAMES:
            threshold = selection['models'][name]['threshold']
            pred = probabilities[name] >= threshold
            totals[name] += np.array([(pred & target).sum(), (pred & ~target).sum(),
                                      (~pred & target).sum()], dtype=np.int64)
            skeleton_hits[name] += int((pred & skeleton).sum())
            baseline_missing_hits[name] += int((pred & baseline_missing).sum())
            if not cv2.imwrite(str(output / name / 'surface' / f'{case}_pred.png'),
                               pred.astype(np.uint8) * 255):
                raise OSError(f'Could not write prediction for {case}, {name}')
        count += 1
        print(f'Test {count}: {case}', flush=True)
    rows = []
    for name in ALL_NAMES:
        row = {'name': name, 'threshold': selection['models'][name]['threshold'],
               'images': count, **scores(*totals[name]),
               'gt_skeleton_coverage_micro': skeleton_hits[name] / skeleton_total if skeleton_total else 0.0,
               'baseline_missed_gt_skeleton_recovered_fraction': (
                   baseline_missing_hits[name] / baseline_missing_total if baseline_missing_total else 0.0)}
        rows.append(row)
    write_csv(args.run_dir / 'test_probe_summary.csv', rows)
    write_json(args.run_dir / 'test_manifest.json', {
        'data_root': str(args.root_path),
        'protocol': {'image_size': NATIVE, 'short_area_threshold': 20,
                     'apls_max_nodes': 64, 'apls_snap_radius': 5.0},
        'models': [{'name': name, 'provenance': {
            'backbone': str(args.model_path), 'backbone_sha256': backbone_hash,
            'probe_checkpoint': str(args.run_dir / 'probe_best.pth'),
            'best_probe_epoch': None if name == 'original_surface' else probe_payload['best_epoch'][name],
            'probe_type': 'frozen-backbone-1x1-road-head' if name != 'original_surface' else 'original-surface',
            'merge': 'native1500 tile512 stride256 weighted logits'},
            'validation_selection': {'selected_on': 'val', 'criterion': 'global_iou',
                                     'threshold': selection['models'][name]['threshold'],
                                     'evidence': str(selected_path)},
            'predictions': {'test': [{'threshold': selection['models'][name]['threshold'],
                                     'path': str(output / name / 'surface')}]}}
            for name in ALL_NAMES]})
    print((args.run_dir / 'test_probe_summary.csv').read_text(encoding='utf-8'), flush=True)


def pixel_diagnostic(args, tap, heads, probe_payload, backbone_hash, device):
    """Pair E1/H0 on current-checkpoint baseline and seven-probe missed pixels."""
    selected_path = args.run_dir / 'val_selection.json'
    if not selected_path.is_file():
        raise FileNotFoundError('Run sweep first: val_selection.json is missing')
    selection = json.loads(selected_path.read_text(encoding='utf-8'))
    if (selection.get('protocol_version') != PROTOCOL_VERSION
            or selection.get('selected_on') != 'val'
            or selection.get('criterion') != 'global_iou'
            or selection.get('backbone_sha256') != backbone_hash):
        raise ValueError('Validation selection belongs to another protocol or checkpoint')
    if set(selection['models']) != set(ALL_NAMES):
        raise ValueError('Validation selection is incomplete')
    if (selection.get('tile_size') != SIZE or selection.get('stride') != STRIDE
            or selection.get('merge') != 'weighted logits then sigmoid'):
        raise ValueError('Validation selection uses a different native-image protocol')
    if probe_payload.get('backbone_sha256') != backbone_hash:
        raise ValueError('Probe heads belong to another backbone')
    thresholds = {name: float(selection['models'][name]['threshold']) for name in ALL_NAMES}
    if any(not 0.0 < value < 1.0 for value in thresholds.values()):
        raise ValueError('Invalid validation-selected threshold')
    if args.expected_missed_pixels is not None:
        print('[WARN] --expected_missed_pixels is deprecated and ignored: '
              '55,865 belongs to the earlier 1af9580 seven-probe checkpoint.', flush=True)

    def rate(numerator, denominator):
        return numerator / denominator if denominator else None

    def mask_hash(mask):
        packed = np.packbits(np.asarray(mask, dtype=np.uint8).ravel(), bitorder='little')
        return hashlib.sha256(
            np.asarray(mask.shape, dtype='<i8').tobytes() + packed.tobytes()
        ).hexdigest()

    def coordinates_hash(indices):
        return hashlib.sha256(
            np.asarray(indices, dtype='<i8').tobytes()
        ).hexdigest()

    def score_stats(scores, prefix):
        return {
            f'{prefix}_prob_mean': float(scores.mean()) if scores.size else None,
            f'{prefix}_prob_p25': float(np.quantile(scores, .25)) if scores.size else None,
            f'{prefix}_prob_p50': float(np.quantile(scores, .50)) if scores.size else None,
            f'{prefix}_prob_p75': float(np.quantile(scores, .75)) if scores.size else None,
        }

    def ast_function_hash(path, function_name):
        if not path.is_file():
            return None
        import ast
        module = ast.parse(path.read_text(encoding='utf-8'))
        matches = [node for node in module.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == function_name]
        if len(matches) != 1:
            return None
        return hashlib.sha256(
            ast.dump(matches[0], include_attributes=False).encode('utf-8')
        ).hexdigest()

    def ast_method_hash(path, class_name, method_name):
        if not path.is_file():
            return None
        import ast
        module = ast.parse(path.read_text(encoding='utf-8'))
        classes = [node for node in module.body
                   if isinstance(node, ast.ClassDef) and node.name == class_name]
        if len(classes) != 1:
            return None
        methods = [node for node in classes[0].body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == method_name]
        if len(methods) != 1:
            return None
        return hashlib.sha256(
            ast.dump(methods[0], include_attributes=False).encode('utf-8')
        ).hexdigest()

    from evaluation.road_comparison import road_metrics as metrics_module
    skeletonize = metrics_module.skeletonize
    names = ('E1', 'H0')
    set_names = ('baseline_missed', 'seven_probe_all_missed')
    accum = {
        set_name: {'pixels': 0, 'e1': 0, 'h0': 0, 'both': 0,
                   'only_e1': 0, 'only_h0': 0, 'neither': 0,
                   'scores': {name: [] for name in names}}
        for set_name in set_names
    }
    background_pixels = 0
    background_predicted = {name: 0 for name in names}
    rows, image_checks, ordered_names = [], [], []
    seen_names = set()
    for case, rgb, target in native_cases(args.root_path, 'test'):
        if case in seen_names:
            raise ValueError(f'Duplicate test image stem: {case}')
        seen_names.add(case)
        ordered_names.append(case)
        probabilities = native_probabilities(tap, heads, rgb, device)
        if set(probabilities) != set(ALL_NAMES):
            raise RuntimeError(f'Missing native probability maps for {case}')
        if any(probability.shape != target.shape or not np.isfinite(probability).all()
               for probability in probabilities.values()):
            raise RuntimeError(f'Unaligned or non-finite native probability map for {case}')
        skeleton = skeletonize(target)
        if skeleton.shape != target.shape or np.any(skeleton & ~target):
            raise RuntimeError(f'GT skeleton is not aligned with the road mask for {case}')
        baseline_missed = skeleton & (probabilities['original_surface'] < thresholds['original_surface'])
        seven_probe_all_missed = baseline_missed.copy()
        seven_missed_on_baseline = {}
        for name in SEVEN_PROBES:
            missed_by_probe = probabilities[name] < thresholds[name]
            seven_probe_all_missed &= missed_by_probe
            seven_missed_on_baseline[name] = int((baseline_missed & missed_by_probe).sum())
        if np.any(seven_probe_all_missed & ~baseline_missed):
            raise RuntimeError(f'Seven-probe set escaped baseline-missed mask for {case}')
        background = ~target
        background_count = int(background.sum())
        background_pixels += background_count
        image_background = {}
        for name in names:
            count = int(((probabilities[name] >= thresholds[name]) & background).sum())
            background_predicted[name] += count
            image_background[name] = count
        masks = {'baseline_missed': baseline_missed,
                 'seven_probe_all_missed': seven_probe_all_missed}
        checks = {
            'image': case,
            'native_shape': list(target.shape),
            'gt_road_mask_sha256': mask_hash(target),
            'gt_skeleton_mask_sha256': mask_hash(skeleton),
            'baseline_missed_pixels': int(baseline_missed.sum()),
            'seven_probe_all_missed_pixels': int(seven_probe_all_missed.sum()),
            'seven_probe_all_missed_subset_of_baseline': True,
            'seven_probe_missed_pixels_within_baseline': seven_missed_on_baseline,
            'all_probability_maps_aligned_to_native_gt': True,
        }
        for set_name, mask in masks.items():
            indices = np.flatnonzero(mask)
            n = int(indices.size)
            coordinate_hash = coordinates_hash(indices)
            checks[f'{set_name}_coordinate_sha256'] = coordinate_hash
            checks[f'{set_name}_mask_sha256'] = mask_hash(mask)
            e1_scores = probabilities['E1'].ravel()[indices].astype(np.float64, copy=False)
            h0_scores = probabilities['H0'].ravel()[indices].astype(np.float64, copy=False)
            e1_hits = e1_scores >= thresholds['E1']
            h0_hits = h0_scores >= thresholds['H0']
            counts = {
                'e1': int(e1_hits.sum()),
                'h0': int(h0_hits.sum()),
                'both': int((e1_hits & h0_hits).sum()),
                'only_e1': int((e1_hits & ~h0_hits).sum()),
                'only_h0': int((~e1_hits & h0_hits).sum()),
                'neither': int((~e1_hits & ~h0_hits).sum()),
            }
            if (counts['both'] + counts['only_e1'] + counts['only_h0']
                    + counts['neither'] != n
                    or counts['e1'] != counts['both'] + counts['only_e1']
                    or counts['h0'] != counts['both'] + counts['only_h0']):
                raise RuntimeError(f'Paired pixel partition failed for {case}, {set_name}')
            slot = accum[set_name]
            slot['pixels'] += n
            for key, value in counts.items():
                slot[key] += value
            slot['scores']['E1'].append(e1_scores.astype(np.float32, copy=False))
            slot['scores']['H0'].append(h0_scores.astype(np.float32, copy=False))
            rows.append({
                'image': case, 'pixel_set': set_name, 'pixels': n,
                'pixel_coordinate_sha256': coordinate_hash,
                'e1_h0_used_identical_pixel_coordinates': True,
                'e1_threshold_selected_on_val': thresholds['E1'],
                'h0_threshold_selected_on_val': thresholds['H0'],
                **{f'{key}_pixels': value for key, value in counts.items()},
                **{f'{key}_fraction': rate(value, n) for key, value in counts.items()},
                **score_stats(e1_scores, 'e1'),
                **score_stats(h0_scores, 'h0'),
                'gt_background_pixels': background_count,
                'e1_gt_background_predicted_road': image_background['E1'],
                'h0_gt_background_predicted_road': image_background['H0'],
                'e1_gt_background_false_positive_rate': rate(image_background['E1'], background_count),
                'h0_gt_background_false_positive_rate': rate(image_background['H0'], background_count),
            })
        image_checks.append(checks)
        print(f'Pixel diagnostic {len(ordered_names)}: {case} | '
              f'baseline={checks["baseline_missed_pixels"]} '
              f'seven_all_missed={checks["seven_probe_all_missed_pixels"]}', flush=True)
    if not ordered_names:
        raise RuntimeError('No test images found')

    summaries = {}
    for set_name in set_names:
        slot = accum[set_name]
        n = slot['pixels']
        e1_scores = np.concatenate(slot['scores']['E1'])
        h0_scores = np.concatenate(slot['scores']['H0'])
        summary = {
            'pixels': n,
            **{f'{key}_pixels': slot[key] for key in
               ('e1', 'h0', 'both', 'only_e1', 'only_h0', 'neither')},
            **{f'{key}_fraction': rate(slot[key], n) for key in
               ('e1', 'h0', 'both', 'only_e1', 'only_h0', 'neither')},
            **score_stats(e1_scores, 'e1'),
            **score_stats(h0_scores, 'h0'),
            'gt_background_pixels': background_pixels,
            'e1_gt_background_predicted_road': background_predicted['E1'],
            'h0_gt_background_predicted_road': background_predicted['H0'],
            'e1_gt_background_false_positive_rate': rate(background_predicted['E1'], background_pixels),
            'h0_gt_background_false_positive_rate': rate(background_predicted['H0'], background_pixels),
        }
        summaries[set_name] = summary
        rows.append({'image': '__micro_total__', 'pixel_set': set_name,
                     'e1_threshold_selected_on_val': thresholds['E1'],
                     'h0_threshold_selected_on_val': thresholds['H0'], **summary})
    if summaries['seven_probe_all_missed']['pixels'] > summaries['baseline_missed']['pixels']:
        raise RuntimeError('Seven-probe missed count exceeds baseline missed count')
    output_csv = args.run_dir / 'test_e1_h0_missed_pixel_diagnostics.csv'
    write_csv(output_csv, rows)

    old_run = args.historical_seven_probe_run
    old_selection_path = old_run / 'val_selection.json' if old_run else None
    old_selection = (json.loads(old_selection_path.read_text(encoding='utf-8'))
                     if old_selection_path and old_selection_path.is_file() else None)
    old_backbone_path = args.historical_backbone_path
    old_backbone_sha = (file_sha256(old_backbone_path)
                        if old_backbone_path and old_backbone_path.is_file() else None)
    old_selection_sha = old_selection.get('backbone_sha256') if old_selection else None
    old_prediction_dir = (old_run / 'test_predictions' / 'original_surface' / 'surface'
                          if old_run else None)
    old_image_names = (sorted(path.stem.removesuffix('_pred') for path in
                              old_prediction_dir.glob('*_pred.png'))
                       if old_prediction_dir and old_prediction_dir.is_dir() else None)
    old_repo = args.historical_seven_probe_repo
    current_script = Path(__file__).resolve()
    current_metrics = Path(metrics_module.__file__).resolve()
    current_dataset = args.probe_repo / 'datasets' / 'dataset_road_skeleton.py'
    current_protocol = args.probe_repo / 'h0_encoder_protocol.py'
    old_script = old_repo / 'probe_experiment.py' if old_repo else None
    old_metrics = old_repo / 'evaluation' / 'road_comparison' / 'road_metrics.py' if old_repo else None
    old_dataset = old_repo / 'datasets' / 'dataset_road_skeleton.py' if old_repo else None
    old_protocol = old_repo / 'h0_encoder_protocol.py' if old_repo else None
    source_checks = {
        'native_case_reader_same_ast': (
            ast_function_hash(current_script, 'native_cases')
            == ast_function_hash(old_script, 'native_cases')
            if old_script and old_script.is_file() else None),
        'native_stitcher_same_ast': (
            ast_function_hash(current_script, 'native_probabilities')
            == ast_function_hash(old_script, 'native_probabilities')
            if old_script and old_script.is_file() else None),
        'gt_skeleton_source_same_sha256': (
            file_sha256(current_metrics) == file_sha256(old_metrics)
            if old_metrics and old_metrics.is_file() else None),
        'gt_skeleton_function_same_ast': (
            ast_function_hash(current_metrics, 'skeletonize')
            == ast_function_hash(old_metrics, 'skeletonize')
            if old_metrics and old_metrics.is_file() else None),
        'dataset_source_same_sha256': (
            file_sha256(current_dataset) == file_sha256(old_dataset)
            if old_dataset and old_dataset.is_file() else None),
        'native_center_crop_or_pad_same_ast': (
            ast_method_hash(current_dataset, 'RoadSkeletonDataset', '_center_crop_or_pad')
            == ast_method_hash(old_dataset, 'RoadSkeletonDataset', '_center_crop_or_pad')
            if old_dataset and old_dataset.is_file() else None),
        'native_label_lookup_same_ast': (
            ast_method_hash(current_dataset, 'RoadSkeletonDataset', '_find_label_name')
            == ast_method_hash(old_dataset, 'RoadSkeletonDataset', '_find_label_name')
            if old_dataset and old_dataset.is_file() else None),
        'blend_weight_same_ast': (
            ast_function_hash(current_protocol, 'blend_weight')
            == ast_function_hash(old_protocol, 'blend_weight')
            if old_protocol and old_protocol.is_file() else None),
        'sliding_positions_same_ast': (
            ast_function_hash(current_protocol, 'sliding_positions')
            == ast_function_hash(old_protocol, 'sliding_positions')
            if old_protocol and old_protocol.is_file() else None),
    }
    old_threshold = (float(old_selection['models']['original_surface']['threshold'])
                     if old_selection and 'original_surface' in old_selection.get('models', {})
                     else 0.5)
    count_delta = summaries['baseline_missed']['pixels'] - 99885
    count_explanation = (
        'The earlier 99,885 baseline-missed pixels and 55,865 seven-probe-all-missed '
        'pixels came from the 1af9580 backbone and its seven probe heads. The current '
        'a400 backbone and ten probe heads use a different checkpoint. '
        'The current baseline count differs by ' + str(count_delta) + ' pixels. '
        + (f'The original_surface threshold also differs from the historical {old_threshold:.2f}.'
           if thresholds['original_surface'] != old_threshold else
           f'The original_surface threshold matches the historical {old_threshold:.2f}. '
           'Different model weights or architecture can change which skeleton '
           'pixels fall below that threshold. The exact per-pixel cause cannot '
           'be partitioned without paired old probability maps.')
    )
    report = {
        'schema': 2,
        'split': 'test',
        'definitions': {
            'baseline_missed': 'Zhang-Suen skeleton(GT road) AND current original_surface probability < its validation-selected threshold',
            'seven_probe_all_missed': 'baseline_missed AND E2, E3, H_E2, H_E3, F2, F3, surface_prehead probabilities each below their own validation-selected threshold',
            'recognized': 'probe probability >= its own validation-selected threshold',
            'background_false_positive_rate': 'predicted road pixels over all GT background pixels; independent of which missed set is evaluated',
        },
        'current_provenance': {
            'architecture_id': MODEL_NAME,
            'expected_source_commit': 'a4001b3dada8006b35fec5d119ea8a382a2134c7',
            'probe_source_path': str(args.probe_repo),
            'probe_source_config_sha256': file_sha256(args.probe_repo / 'h0_e2_config.py'),
            'backbone_path': str(args.model_path),
            'backbone_sha256': backbone_hash,
            'probe_checkpoint': str(args.run_dir / 'probe_best.pth'),
            'probe_checkpoint_sha256': file_sha256(args.run_dir / 'probe_best.pth'),
            'probe_checkpoint_backbone_sha256': probe_payload['backbone_sha256'],
            'validation_selection_path': str(selected_path),
            'validation_selection_sha256': file_sha256(selected_path),
            'thresholds_selected_on_val': thresholds,
            'validation_criterion': selection['criterion'],
            'ordered_test_images': ordered_names,
            'ordered_test_images_sha256': hashlib.sha256(
                json.dumps(ordered_names, ensure_ascii=False).encode('utf-8')
            ).hexdigest(),
            'test_images': len(ordered_names),
            'gt_road_threshold': 'grayscale > 127',
            'gt_skeleton': 'evaluation.road_comparison.road_metrics.skeletonize; NumPy Zhang-Suen with zero padding',
            'gt_skeleton_source_sha256': file_sha256(current_metrics),
            'native_size': NATIVE,
            'tile_size': SIZE,
            'stride': STRIDE,
            'tile_merge': 'weighted logits then sigmoid',
            'blend_weight': 'outer product of 1 - abs(linspace(-1,1,512)), clamped at 0.1',
        },
        'historical_seven_probe_comparison': {
            'source_backbone_commit': '1af958056f5953f0eb6edbffcdb9f1a9db77ba63',
            'historical_selection_protocol_version': (
                old_selection.get('protocol_version') if old_selection else None),
            'historical_selection_criterion': (
                old_selection.get('criterion') if old_selection else None),
            'historical_backbone_path': str(old_backbone_path) if old_backbone_path else None,
            'historical_backbone_sha256': old_backbone_sha,
            'historical_selection_backbone_sha256': old_selection_sha,
            'historical_backbone_file_matches_its_selection': (
                old_backbone_sha == old_selection_sha
                if old_backbone_sha and old_selection_sha else None),
            'same_backbone_sha256_as_current': (
                old_selection_sha == backbone_hash
                if old_selection_sha else
                (old_backbone_sha == backbone_hash if old_backbone_sha else None)),
            'historical_test_images': 49,
            'current_test_image_count_matches_historical': len(ordered_names) == 49,
            'historical_baseline_missed_pixels': 99885,
            'historical_seven_probe_all_missed_pixels': 55865,
            'historical_surface_threshold': old_threshold,
            'current_surface_threshold_matches_historical': thresholds['original_surface'] == old_threshold,
            'historical_test_prediction_images_available': old_image_names is not None,
            'same_ordered_test_image_names_as_historical_predictions': (
                old_image_names == ordered_names
                if old_image_names is not None else None),
            'historical_image_order_hash_available': False,
            'historical_gt_skeleton_coordinate_hashes_available': False,
            'image_order_rule': 'Both scripts iterate sorted RoadSkeletonDataset.image_files',
            'source_checks': source_checks,
            'current_baseline_minus_historical_baseline_pixels': count_delta,
            'count_difference_explanation': count_explanation,
            'historical_55865_is_not_an_expected_current_count': True,
        },
        'coordinate_checks': {
            'all_probability_maps_match_native_gt_shape': True,
            'all_seven_probe_all_missed_pixels_are_within_baseline_missed': True,
            'e1_and_h0_use_the_same_flat_pixel_indices_for_each_set': True,
            'pixel_index_order': 'row-major within each native 1500x1500 image',
            'per_image_mask_and_coordinate_hashes': image_checks,
        },
        'summary': summaries,
        'per_image_csv': str(output_csv),
        'interpretation_limit': 'Frozen one-layer probe detections indicate readable evidence at fixed validation thresholds; they do not prove that the original decoder could use it or that changing the model would improve connectivity.',
    }
    output_json = args.run_dir / 'test_missed_pixel_set.json'
    write_json(output_json, report)
    print(json.dumps({'current_baseline_missed': summaries['baseline_missed']['pixels'],
                      'current_seven_probe_all_missed': summaries['seven_probe_all_missed']['pixels'],
                      'seven_all_missed_e1': summaries['seven_probe_all_missed']['e1_pixels'],
                      'seven_all_missed_h0': summaries['seven_probe_all_missed']['h0_pixels'],
                      'report': str(output_json)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe_repo', type=Path, required=True)
    parser.add_argument('mode', choices=('train', 'resume', 'sweep', 'test', 'pixel_diagnostic'))
    parser.add_argument('--root_path', type=Path, required=True)
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_train_images', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--historical_seven_probe_run', type=Path,
                        help='optional 1af9580 seven-probe run for validation-selection and test-image audit')
    parser.add_argument('--historical_backbone_path', type=Path,
                        help='optional 1af9580 best.pth for exact SHA256 comparison')
    parser.add_argument('--historical_seven_probe_repo', type=Path,
                        help='optional 9432ac1 seven-probe source checkout for protocol source comparison')
    parser.add_argument('--expected_missed_pixels', type=int, default=None,
                        help='deprecated and ignored; historical 55,865 is from another checkpoint')
    args = parser.parse_args()
    if args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error('Epochs, patience, and batch size must be positive; workers nonnegative')
    if args.max_train_images < 1 or not 0 < args.lr < 1:
        parser.error('max_train_images must be positive and lr must be in (0,1)')
    args.model_path = args.model_path.resolve(strict=True)
    args.root_path = args.root_path.resolve(strict=True)
    args.run_dir = args.run_dir.resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        parser.error('The 512x512 H0 probe experiment requires CUDA')
    device = torch.device('cuda:0')
    backbone_hash = file_sha256(args.model_path)
    model = load_backbone(args.model_path, device)
    tap = FeatureTap(model)
    try:
        if args.mode in ('train', 'resume'):
            train(args, model, tap, backbone_hash, device)
        else:
            heads, probe_payload = load_heads(args.run_dir / 'probe_best.pth', backbone_hash, device)
            if args.mode == 'sweep':
                sweep(args, tap, heads, backbone_hash, device)
            elif args.mode == 'pixel_diagnostic':
                pixel_diagnostic(args, tap, heads, probe_payload, backbone_hash, device)
            else:
                test(args, tap, heads, probe_payload, backbone_hash, device)
    finally:
        tap.close()


if __name__ == '__main__':
    main()
