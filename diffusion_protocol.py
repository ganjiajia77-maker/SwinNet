"""Architecture contract and explicit H0 shared-parameter initialization."""
import hashlib
import json
from pathlib import Path
import torch
from config import _C
from networks.directional_diffusion import ARCHITECTURE_ID, MODES
from networks.vision_transformer import SwinUnet

BASE_COMMIT = '599a410de8d341850e1810417307e4b3bbdfc82d'
RUNTIME_BASE = 'fe3b73098238bd7a191f6e0b5ce458488cdd394f'
ROOT = Path(__file__).resolve().parent


def architecture(mode='semantic_directional', steps=8, eta=.1, beta=.5, latent=32):
    if mode not in MODES:
        raise ValueError('Unknown diffusion control mode')
    return dict(architecture_id=ARCHITECTURE_ID, diffusion_mode=mode, diffusion_steps=int(steps),
                diffusion_eta=float(eta), diffusion_beta=float(beta), diffusion_latent_channels=int(latent),
                stage2_heads={'S': True, 'C': False, 'D': False}, stage2_direct_H_injection=False,
                stage3_heads={'S': True, 'C': False, 'D': True}, stage3_old_gate=False,
                global_topology=False, highres_structure_channels=64, enable_h3_surface_fusion=True,
                skeleton_mode='direct', base_h0_commit=BASE_COMMIT)


def config(pretrain=None):
    cfg = _C.clone()
    cfg.defrost()
    cfg.merge_from_file(str(ROOT/'configs/swin_tiny_patch4_window7_224_lite.yaml'))
    cfg.DATA.IMG_SIZE = 512
    cfg.MODEL.PRETRAIN_CKPT = str(pretrain) if pretrain else None
    cfg.freeze()
    return cfg


def build_model(arc, pretrain=None, original=False):
    if not original:
        expected = architecture(arc['diffusion_mode'], arc['diffusion_steps'], arc['diffusion_eta'],
                                arc['diffusion_beta'], arc['diffusion_latent_channels'])
        if arc != expected:
            raise ValueError('Architecture contract mismatch')
    kwargs = {} if original else {k: arc[k] for k in (
        'architecture_id', 'diffusion_mode', 'diffusion_steps', 'diffusion_eta',
        'diffusion_beta', 'diffusion_latent_channels')}
    return SwinUnet(config(pretrain), img_size=512, num_classes=1, return_skeleton=True,
        structure_profile='stage23_boundary_0626', enable_highres_structure_stream=True,
        highres_structure_channels=64, highres_structure_fuse_stages='stage23',
        highres_structure_fusion_mode='stage23', enable_h3_surface_fusion=True,
        enable_global_topology=original, stage_skeleton_mode='direct',
        stage2_skeleton_gradient_ratio=.5, stage3_skeleton_gradient_ratio=.5,
        stage3_gate_topology_gradient_ratio=0., remove_stage2_pre_topology_source=True, **kwargs)


def initialize_from_h0(model, arc, seed, pretrain=None, h0_checkpoint=None):
    # A separate original H0 reference makes shared initialization independent of
    # constructor RNG consumption in the smaller architecture and all 3 modes.
    print('[INITIALIZATION ONLY] Building a separate original H0 reference; it is not the training model.', flush=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = build_model(arc, pretrain, original=True)
        loaded_names = set(reference.load_from(config(pretrain))) if pretrain else set()
    source = reference.state_dict()
    if h0_checkpoint:
        checkpoint = torch.load(h0_checkpoint, map_location='cpu', weights_only=False)
        candidate = checkpoint['model_state_dict']
        if set(candidate) != set(source) or any(candidate[k].shape != source[k].shape for k in source):
            raise ValueError('Explicit H0 initialization checkpoint is not the original configured H0')
        source = candidate
        loaded_names = set(checkpoint.get('loaded_pretrained_names', loaded_names))
    target = model.state_dict()
    shared = sorted(set(source) & set(target))
    wrong = [key for key in shared if source[key].shape != target[key].shape]
    if wrong:
        raise ValueError(f'Shared H0 tensor shapes changed: {wrong}')
    added = sorted(set(target)-set(source))
    if any('.diffusion.' not in key for key in added):
        raise ValueError(f'Unexpected new parameter/buffer names: {added}')
    for key in shared:
        target[key] = source[key]
    model.load_state_dict(target, strict=True)
    retained = model.state_dict()
    if any(not torch.equal(retained[key].cpu(), source[key].cpu()) for key in shared):
        raise RuntimeError('Shared H0 tensor initialization comparison failed')
    digest = hashlib.sha256()
    for key in shared:
        digest.update(key.encode())
        digest.update(source[key].cpu().contiguous().numpy().tobytes())
    manifest = dict(base_commit=BASE_COMMIT, runtime_base_commit=RUNTIME_BASE, seed=seed,
        source=str(h0_checkpoint) if h0_checkpoint else 'seeded original H0 + same ImageNet encoder pretraining',
        shared_tensors=shared, deleted_tensors=sorted(set(source)-set(target)), new_tensors=added,
        shared_tensors_equal=True, shared_tensor_sha256=digest.hexdigest(),
        original_parameters=sum(p.numel() for p in reference.parameters()),
        new_parameters=sum(p.numel() for p in model.parameters()))
    print(f'[INITIALIZATION COMPLETE] {len(shared)} shared tensors compared equal; '
          f'{len(manifest["deleted_tensors"])} deleted; {len(added)} new. Training model has no C2/D2/C3/global nodes.', flush=True)
    return loaded_names & set(dict(model.named_parameters())), manifest


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('architecture_id') != ARCHITECTURE_ID or checkpoint.get('base_h0_commit') != BASE_COMMIT:
        raise ValueError('This entry requires the new Stage2-skeleton/Stage3-diffusion checkpoint')
    arc = checkpoint['architecture_config']
    model = build_model(arc)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if checkpoint.get('args', {}).get('use_ema') is not False:
        raise ValueError('The selected Massachusetts experiment uses no EMA')
    return model, checkpoint


def save_atomic(checkpoint, path):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
