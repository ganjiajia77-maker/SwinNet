"""Read-only regional inspection of an original H0 Mass checkpoint."""
import argparse
import csv
import hashlib
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from compare_connectivity_topology_metrics import load_metric_model, parse_args as model_parser
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from topology_direction_constants import CONNECTIVITY_DIRECTIONS
from losses.road_losses import build_stage_skeleton_target, build_connectivity_target


def shifted(a, dy, dx):
    out = np.zeros_like(a)
    h, w = a.shape
    y0, y1 = max(0, -dy), min(h, h-dy)
    x0, x1 = max(0, -dx), min(w, w-dx)
    out[y0:y1, x0:x1] = a[y0+dy:y1+dy, x0+dx:x1+dx]
    return out


def region_masks(gt, skeleton, pred, gap_distance):
    # Opposite nearby road pixels identify a background corridor candidate, not a GT connection.
    road = gt.astype(np.uint8)
    near = cv2.dilate(road, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    corridor = np.zeros_like(gt, dtype=bool)
    for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
        for r in range(1, gap_distance + 1):
            corridor |= shifted(near, dy*r, dx*r) & shifted(near, -dy*r, -dx*r)
    return {'skeleton_center': skeleton, 'road_noncenter': gt & ~skeleton,
            'candidate_between_roads': ~gt & corridor,
            'background_far': ~gt & ~near, 'missed_road': gt & ~pred,
            'missed_gt_skeleton': skeleton & ~pred}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


class Capture:
    def __init__(self, model):
        self.net = model.swin_unet
        self.mode = 'on'
        self.maps = {}
        self.handles = []
        if not self.net.enable_highres_structure_stream or not self.net.enable_global_topology:
            raise ValueError('Requires original H0 with high-resolution stream and global topology enabled')
        for s in ('2', '3'):
            self.handles.append(self.net.highres_structure_fusion[s].register_forward_hook(self.fusion(s)))
            self.handles.append(self.net.decoder_structure_blocks[s].register_forward_hook(self.stage(s)))
        self.original_global = self.net.global_topology.forward_feature_anchors
        self.net.global_topology.forward_feature_anchors = types.MethodType(self.global_forward, self.net.global_topology)

    def fusion(self, s):
        def hook(module, inputs, output):
            if self.mode == 'on':
                self.maps['h_injection_stage'+s] = (output-inputs[0]).abs().mean(1, keepdim=True)
                self.maps['h_input_stage'+s] = inputs[0].abs().mean(1, keepdim=True)
            if self.mode == 'h_direct_off':
                return inputs[0]
        return hook

    def stage(self, s):
        def hook(module, inputs, output):
            if self.mode == 'on':
                self.maps['c'+s] = output[2].sigmoid()
                self.maps['stage_gate'+s] = output[4]
        return hook

    def global_forward(self, module, feature, z_struct, surface_prob, connectivity_feature=None):
        if self.mode == 'global_off':
            return feature
        output = self.original_global(feature, z_struct, surface_prob, connectivity_feature=connectivity_feature)
        if self.mode == 'on':
            self.maps['global_feature_delta'] = (output-feature).abs().mean(1, keepdim=True)
            self.maps['global_feature_input'] = feature.abs().mean(1, keepdim=True)
            self.maps['global_surface_gate'] = surface_prob
        return output

    def run(self, model, image, mode):
        self.mode = mode
        self.maps = {}
        result = model(image)
        return result[0] if isinstance(result, tuple) else result

    def close(self):
        for h in self.handles:
            h.remove()
        self.net.global_topology.forward_feature_anchors = self.original_global


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root_path', required=True)
    p.add_argument('--model_path', required=True)
    p.add_argument('--cfg', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--threshold', type=float, required=True)
    p.add_argument('--max_images', type=int, default=0)
    p.add_argument('--gap_distance', type=int, default=16)
    a = p.parse_args()
    if not (0 < a.threshold < 1) or a.max_images < 0 or not 1 <= a.gap_distance <= 100:
        p.error('invalid threshold, max_images, or gap_distance')
    dst = Path(a.output_dir)
    if dst.exists() and any(dst.iterdir()):
        p.error('output directory must be new or empty')
    dst.mkdir(parents=True, exist_ok=True)
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], '--root_path', a.root_path, '--model_path', a.model_path,
                    '--cfg', a.cfg, '--img_size', '512', '--source_patch_size', '1500']
        margs = model_parser()
    finally:
        sys.argv = old_argv
    # The legacy model parser omits this field, although its loader reads it.
    margs.enable_post_refine_structure_interaction = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_metric_model(a.model_path, margs, device).eval()
    dataset = RoadSkeletonDataset(a.root_path, split=a.split, image_size=None, source_patch_size=1500)
    capture = Capture(model)
    parameters = {'stage2_gamma1': float(capture.net.decoder_structure_blocks['2'].gamma1.detach().cpu()),
                  'stage3_gamma1': float(capture.net.decoder_structure_blocks['3'].gamma1.detach().cpu()),
                  'alpha_global': float(capture.net.global_topology.alpha_global.detach().cpu())}
    names = dataset.image_files[:a.max_images] if a.max_images else dataset.image_files
    rows = []
    c_stats = {s: defaultdict(float) for s in ('2','3')}
    win = (1-torch.linspace(-1, 1, 512, device=device).abs()).clamp_min(.05)
    weight = torch.outer(win, win)[None, None]
    means = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    try:
        with torch.inference_mode():
            for index, name in enumerate(names, 1):
                im = cv2.imread(str(Path(dataset.image_dir)/name), cv2.IMREAD_COLOR)
                label_name = dataset._find_label_name(name, dataset.mask_dir)
                gt_img = cv2.imread(str(Path(dataset.mask_dir)/label_name), cv2.IMREAD_GRAYSCALE)
                if im is None or gt_img is None or im.shape[:2] != (1500, 1500) or gt_img.shape != (1500, 1500):
                    raise ValueError(f'{name}: expected readable paired native 1500 images')
                rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                gt = gt_img > 127
                skeleton = dataset._skeletonize_binary(gt.astype(np.uint8)*255) > 127
                canvases = {}
                denominator = torch.zeros((1, 1, 1500, 1500), device=device)
                for y in dataset.sliding_positions(1500, 512, 256):
                    for x in dataset.sliding_positions(1500, 512, 256):
                        tile = torch.from_numpy(rgb[y:y+512, x:x+512].copy()).to(device).permute(2, 0, 1)[None].float()/255
                        tile = (tile-means)/std
                        on = capture.run(model, tile, 'on')
                        maps = dict(capture.maps)
                        sk_tile = torch.from_numpy(skeleton[y:y+512, x:x+512].copy()).to(device)[None,None].float()
                        for s in ('2','3'):
                            probability = maps['c'+s]
                            sk = build_stage_skeleton_target(sk_tile, probability.shape[-2:]) > .5
                            target = build_connectivity_target(sk.float()) > .5
                            valid = sk.expand_as(target)
                            # Exclude tile borders: neighbors outside the window are unknown.
                            valid = valid.clone()
                            valid[:,:,[0,-1],:] = False
                            valid[:,:,:,[0,-1]] = False
                            for key, mask in (('positive', valid & target), ('negative', valid & ~target)):
                                c_stats[s][key+'_count'] += int(mask.sum().item())
                                c_stats[s][key+'_prob_sum'] += float(probability[mask].sum().item())
                                c_stats[s][key+'_predicted_positive'] += int((probability[mask] >= .5).sum().item())
                        maps['logit_on'] = on
                        maps['logit_global_off'] = capture.run(model, tile, 'global_off')
                        maps['logit_h_direct_off'] = capture.run(model, tile, 'h_direct_off')
                        required = {'c2','c3','stage_gate2','stage_gate3','global_surface_gate','global_feature_delta',
                                    'global_feature_input','h_injection_stage2','h_injection_stage3',
                                    'h_input_stage2','h_input_stage3','logit_on','logit_global_off','logit_h_direct_off'}
                        if not required.issubset(maps):
                            raise RuntimeError(f'missing expected active maps: {sorted(required-set(maps))}')
                        for key, value in maps.items():
                            value = F.interpolate(value.float(), (512,512), mode='bilinear', align_corners=False) if value.shape[-2:] != (512,512) else value.float()
                            if key not in canvases:
                                canvases[key] = torch.zeros((1,value.shape[1],1500,1500), device=device)
                            canvases[key][:,:,y:y+512,x:x+512] += value*weight
                        denominator[:,:,y:y+512,x:x+512] += weight
                maps = {k: (v/denominator.clamp_min(1e-8))[0].cpu().numpy() for k,v in canvases.items()}
                pred = maps['logit_on'][0] >= np.log(a.threshold/(1-a.threshold))
                regions = region_masks(gt, skeleton, pred, a.gap_distance)
                derived = {k:v[0] for k,v in maps.items() if v.shape[0] == 1}
                derived['global_logit_delta'] = maps['logit_on'][0]-maps['logit_global_off'][0]
                derived['h_direct_logit_delta'] = maps['logit_on'][0]-maps['logit_h_direct_off'][0]
                for stage in ('2','3'):
                    derived['c'+stage+'_top2'] = np.sort(maps['c'+stage], axis=0)[-2:].mean(axis=0)
                    for j,direction in enumerate(CONNECTIVITY_DIRECTIONS):
                        derived['c'+stage+'_'+str(j)] = maps['c'+stage][j]
                for region, mask in regions.items():
                    count = int(mask.sum())
                    row = {'image':name, 'region':region, 'pixels':count}
                    for key,v in derived.items():
                        row[key+'_mean'] = float(v[mask].mean()) if count else None
                    rows.append(row)
                print(f'[{index}/{len(names)}] {name} missed_skeleton={regions["missed_gt_skeleton"].sum()}', flush=True)
    finally:
        capture.close()
    if not rows:
        raise RuntimeError('no images')
    with (dst/'per_image_regions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {'checkpoint':str(Path(a.model_path).resolve()), 'checkpoint_sha256':sha256(a.model_path),
               'split':a.split,'images':len(names),'threshold':a.threshold,'parameters':parameters,
               'protocol':'native1500, 512 windows stride256, tapered logit merge; region means pixel weighted',
               'notes':['candidate_between_roads is a geometric background corridor proxy, not verified true gaps',
                        'C channel order N NE E SE S SW W NW; spatial probabilities are interpolated to pixels',
                        'C target audit uses original stage max-pooled GT skeleton and 8-neighbor target at native stage resolution; tile borders excluded, overlapping windows counted repeatedly',
                        'H direct off disables stage2/3 direct H injection only; bidirectional exchange and global H remain',
                        'global_logit_delta and h_direct_logit_delta are paired forward interventions, not additive causal decompositions'],
               'regions':{}}
    summary['connectivity_target_audit'] = {}
    for stage, stats in c_stats.items():
        result = dict(stats)
        for k in ('positive','negative'):
            n = stats[k+'_count']
            result[k+'_prob_mean'] = stats[k+'_prob_sum']/n if n else None
            result[k+'_predicted_positive_rate_at_0.5'] = stats[k+'_predicted_positive']/n if n else None
        summary['connectivity_target_audit']['stage'+stage] = result
    for region in regions:
        subset = [r for r in rows if r['region']==region]
        n = sum(r['pixels'] for r in subset)
        summary['regions'][region] = {'pixels':n}
        for k in rows[0]:
            if k.endswith('_mean'):
                summary['regions'][region][k] = sum(r[k]*r['pixels'] for r in subset if r[k] is not None)/n if n else None
    (dst/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved {dst}', flush=True)


if __name__ == '__main__':
    main()
