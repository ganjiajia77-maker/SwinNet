"""Exact original decoder-location H0 loader and native Massachusetts IO.

No encoder-H installation, Stage2 global relocation, EMA fallback or training.
The constructor and strict checkpoint contract come from baseline 892e5d0.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from h0_location import MODEL_NAMES, validate_reference_args
from networks.h0_location_contract import ARCHITECTURE_ID, BASE_MODEL_COMMIT, load_checkpoint_strict
from networks.vision_transformer import SwinUnet


ROOT = Path(__file__).resolve().parents[1]
SIZE = 1500
NAME = MODEL_NAMES['decoder']
EXPECTED = {'val': 14, 'test': 49}


def discover_checkpoint(root):
    """Find identity-named run pointers/weights; never guess among several runs.

    No tensor files are loaded during discovery. The selected file is validated
    against its full saved architecture by load_model before any forward pass.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Checkpoint search root does not exist: '+str(root))
    found = set()
    pointer_name = ARCHITECTURE_ID+'_decoder_latest.json'
    skip = {'data1', 'Massachusetts', 'tiff', 'datasets', 'miniconda3', 'anaconda3',
            'pretrained_ckpt', 'venv', 'env', '__pycache__', 'node_modules'}
    for directory, children, files in os.walk(root):
        parent = Path(directory)
        depth = len(parent.relative_to(root).parts)
        children[:] = [d for d in children if not d.startswith('.') and d not in skip] if depth < 6 else []
        if pointer_name in files:
            try:
                pointer = json.loads((parent/pointer_name).read_text(encoding='utf-8-sig'))
                if (pointer.get('architecture_id') == ARCHITECTURE_ID
                        and pointer.get('stage2_structure_location') == 'decoder'):
                    target = Path(pointer['best'])
                    if target.is_file():
                        found.add(target.resolve())
            except (OSError, ValueError, KeyError, TypeError):
                pass
        if 'best.pth' in files and any((ARCHITECTURE_ID+'_decoder') in p for p in parent.relative_to(root).parts):
            found.add((parent/'best.pth').resolve())
    if len(found) != 1:
        listing = '\n'.join(str(p) for p in sorted(found)) or '(none)'
        raise ValueError('Expected one original decoder-control best.pth; specify --checkpoint explicitly. '
                         'Candidates:\n'+listing)
    return found.pop()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def tensor_digests(model):
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in model.state_dict().items()}


def counts(pred, gt):
    return int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())


def metrics(tp, fp, fn):
    return dict(iou=tp/max(tp+fp+fn, 1), f1=2*tp/max(2*tp+fp+fn, 1),
                precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1))


def validate_saved(saved, contract, has_ema=False):
    expected = dict(architecture_id=ARCHITECTURE_ID, stage2_structure_location='decoder',
                    base_model_commit=BASE_MODEL_COMMIT)
    if not isinstance(contract, dict) or any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError('Expected original decoder control architecture_contract: '+str(expected))
    if not isinstance(saved, dict):
        raise ValueError('Checkpoint is missing resolved training args')
    if saved.get('architecture_id') != ARCHITECTURE_ID or saved.get('stage2_structure_location') != 'decoder':
        raise ValueError('Expected decoder location experiment; refusing other H0/encoder variants')
    validate_reference_args(saved)
    if has_ema or saved.get('encoder_h_postmerge_phase') or saved.get('global_topology_stage2'):
        raise ValueError('Expected raw no-EMA original decoder control, without encoder-H/Stage2-global relocation')


def build_model(saved):
    options = SimpleNamespace(cfg=str(ROOT/'configs/swin_tiny_patch4_window7_224_lite.yaml'),
        opts=None, batch_size=2, img_size=512, zip=False, cache_mode=None, resume=None,
        accumulation_steps=1, use_checkpoint=False, amp_opt_level=None, tag=None,
        eval=True, throughput=False)
    config = get_config(options)
    parameters = inspect.signature(SwinUnet).parameters
    kwargs = {k: v for k, v in saved.items() if k in parameters
              and k not in ('config', 'img_size', 'num_classes', 'return_skeleton')}
    return SwinUnet(config, img_size=512, num_classes=1, return_skeleton=True, **kwargs)


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    saved = state.get('args')
    validate_saved(saved, state.get('architecture_contract'), state.get('ema_state_dict') is not None)
    model = build_model(saved)
    # Exact contract and every tensor must match; no missing-key allowances.
    load_checkpoint_strict(model, state)
    epoch = state.get('epoch')
    if not isinstance(epoch, int) or isinstance(epoch, bool) or not 1 <= epoch <= 80:
        raise ValueError('Expected checkpoint epoch in 1..80')
    del state
    model.requires_grad_(False)
    print(f'[RESTORE] strict=True; architecture={ARCHITECTURE_ID}; location=decoder; '
          f'weights=model_state_dict; EMA=False; epoch={epoch}', flush=True)
    return model.to(device).eval(), saved, epoch


def dataset(root, split):
    data = RoadSkeletonDataset(str(root), split, image_size=None, source_patch_size=SIZE,
                               augment=False, return_full_image=True)
    if len(data) != EXPECTED[split]:
        raise ValueError(f'Expected Massachusetts {EXPECTED[split]} {split} images, found {len(data)}')
    return data


def read_sample(data, index):
    name = data.image_files[index]
    image = cv2.imread(str(Path(data.image_dir)/name), cv2.IMREAD_COLOR)
    # Replicated RGB Massachusetts labels must really be identical-channel binary GT.
    mask = cv2.imread(str(Path(data.mask_dir)/data.label_files[index]), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        raise IOError('Cannot read native image/GT: '+name)
    if mask.ndim == 3 and mask.shape[2] == 3:
        if not (np.array_equal(mask[:, :, 0], mask[:, :, 1]) and np.array_equal(mask[:, :, 0], mask[:, :, 2])):
            raise ValueError('GT RGB channels disagree: '+name)
        mask = mask[:, :, 0]
    if image.shape != (SIZE, SIZE, 3) or mask.shape != (SIZE, SIZE):
        raise ValueError('Expected native1500 Massachusetts RGB/GT: '+name)
    if not np.isin(np.unique(mask), (0, 255)).all():
        raise ValueError('Expected original binary 0/255 GT: '+name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    image = (image-np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
    return Path(name).stem, torch.from_numpy(image.transpose(2, 0, 1).copy()), mask > 127


def select_validation_record(checkpoint, checkpoint_sha, selection):
    """Accept raw identity-bound records or the original unified selection+manifest.

    Unified selection must name the decoder control and match the manifest's
    checkpoint SHA and model identity. Threshold is never inferred from test masks.
    """
    paths = [selection] if selection else sorted(checkpoint.parent.rglob('val_selection.json'))
    choices = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding='utf-8-sig'))
            if record.get('selected_on') != 'val' or record.get('criterion') != 'global_iou':
                continue
            if isinstance(record.get('models'), dict) and NAME in record['models']:
                entry = record['models'][NAME]
                threshold = float(entry['threshold'])
                manifests = [path.parent/'comparison.json', path.parent.parent/'comparison.json']
                valid = False
                for manifest_path in manifests:
                    if not manifest_path.is_file():
                        continue
                    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
                    for model in manifest.get('models', []):
                        provenance = model.get('provenance', {})
                        identity = hashlib.sha256(json.dumps(dict(name=model['name'], provenance=provenance), sort_keys=True).encode()).hexdigest()
                        candidates = model.get('predictions', {}).get('val', [])
                        if (model['name'] == NAME and provenance.get('checkpoint_sha256') == checkpoint_sha
                                and entry.get('identity') == identity
                                and any(float(c['threshold']) == threshold for c in candidates)):
                            valid = True
                if not valid:
                    continue
            else:
                threshold = float(record['threshold'])
                if (record.get('checkpoint_sha256') != checkpoint_sha
                        or record.get('eval_weights') not in (None, 'raw', 'model_state_dict')):
                    continue
            if np.isfinite(threshold) and 0 < threshold < 1:
                choices.append((path.resolve(), record, threshold))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not choices:
        raise ValueError('No matching decoder-control validation selection found. Pass --selection to its '
            'val_selection.json; unified records also need their sibling comparison.json with checkpoint provenance.')
    if len({round(c[2], 7) for c in choices}) > 1:
        raise ValueError('Multiple val-selected thresholds exist; specify --selection:\n'
                         +'\n'.join(f'{p}: {t}' for p, _, t in choices))
    return max(choices, key=lambda c: c[0].stat().st_mtime_ns)
