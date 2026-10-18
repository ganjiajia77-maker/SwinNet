"""Architecture contract and explicit H0 shared-parameter initialization."""
import hashlib
import json
from pathlib import Path
import torch
from config import _C
from networks.directional_diffusion import ARCHITECTURE_ID, MODES
from networks.encoder_h256 import ARCHITECTURE_ID as H256_ID
from networks.vision_transformer import SwinUnet

BASE_COMMIT = '599a410de8d341850e1810417307e4b3bbdfc82d'
RUNTIME_BASE = 'fe3b73098238bd7a191f6e0b5ce458488cdd394f'
ROOT = Path(__file__).resolve().parent
EXPERIMENT_BASE = 'a6687113860b0bf490c5d578889b47093906ff18'


def architecture(mode='semantic_directional', steps=8, eta=.1, beta=.5, latent=32,
                 architecture_id=ARCHITECTURE_ID, encoder_h_exchange=True, enable_h_surface_residual=True):
    if mode not in MODES:
        raise ValueError('Unknown diffusion control mode')
    result = dict(architecture_id=architecture_id, diffusion_mode=mode, diffusion_steps=int(steps),
                diffusion_eta=float(eta), diffusion_beta=float(beta), diffusion_latent_channels=int(latent),
                stage2_heads={'S': True, 'C': False, 'D': False}, stage2_direct_H_injection=False,
                stage3_heads={'S': True, 'C': False, 'D': True}, stage3_old_gate=False,
                global_topology=False, highres_structure_channels=64, enable_h3_surface_fusion=True,
                skeleton_mode='direct', base_h0_commit=BASE_COMMIT)
    if architecture_id == H256_ID:
        if (int(steps),float(eta),float(beta),int(latent)) != (8,.1,.5,32):
            raise ValueError('H256 protocol fixes diffusion K8/eta0.1/beta0.5/latent32')
        result.update(base_experiment_commit=EXPERIMENT_BASE, highres_grid=256,
            highres_stem_stride=2, stem_second_conv_stride=1, adapter_calls=1,
            encoder_h_exchange=bool(encoder_h_exchange), encoder_exchange_positions='after_swin_before_patchmerging',
            encoder_stage_channels=[96,192,384], encoder_pool_factors=[2,4,8],
            encoder_exchange_initial_strength=.05, A1A2=False, decoder_highres_grid=256,
            diffusion_grid=128, enable_h_surface_residual=bool(enable_h_surface_residual),
            h_surface_projection_channels=16, h_surface_position='after_G3_fusion_before_guided_head')
    elif architecture_id != ARCHITECTURE_ID:
        raise ValueError('Unknown architecture ID')
    return result


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
                                arc['diffusion_beta'], arc['diffusion_latent_channels'], arc['architecture_id'],
                                arc.get('encoder_h_exchange', True), arc.get('enable_h_surface_residual', True))
        if arc != expected:
            raise ValueError('Architecture contract mismatch')
    kwargs = {} if original else {k: arc[k] for k in (
        'architecture_id', 'diffusion_mode', 'diffusion_steps', 'diffusion_eta',
        'diffusion_beta', 'diffusion_latent_channels')}
    if not original and arc['architecture_id'] == H256_ID:
        kwargs.update(encoder_h_exchange=arc['encoder_h_exchange'],
                      enable_h_surface_residual=arc['enable_h_surface_residual'])
    return SwinUnet(config(pretrain), img_size=512, num_classes=1, return_skeleton=True,
        structure_profile='stage23_boundary_0626', enable_highres_structure_stream=True,
        highres_structure_channels=64, highres_structure_fuse_stages='stage23',
        highres_structure_fusion_mode='stage23', enable_h3_surface_fusion=True,
        enable_global_topology=original, stage_skeleton_mode='direct',
        stage2_skeleton_gradient_ratio=.5, stage3_skeleton_gradient_ratio=.5,
        stage3_gate_topology_gradient_ratio=0., remove_stage2_pre_topology_source=True, **kwargs)


def initialize_from_h0(model, arc, seed, pretrain=None, h0_checkpoint=None, base_checkpoint=None):
    if arc['architecture_id'] == H256_ID:
        old_arc = architecture(arc['diffusion_mode'], arc['diffusion_steps'], arc['diffusion_eta'],
                               arc['diffusion_beta'], arc['diffusion_latent_channels'])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            reference = build_model(old_arc)
            loaded_names, original_manifest = initialize_from_h0(reference, old_arc, seed, pretrain, h0_checkpoint)
        source = reference.state_dict()
        if base_checkpoint:
            checkpoint = torch.load(base_checkpoint, map_location='cpu', weights_only=False)
            if checkpoint.get('architecture_config') != old_arc:
                raise ValueError('Explicit migration source must be the matching a668711 architecture')
            candidate = checkpoint['model_state_dict']
            if set(candidate) != set(source) or any(candidate[k].shape != source[k].shape for k in source):
                raise ValueError('Explicit baseline migration state does not match a668711')
            source = candidate
            loaded_names = set(checkpoint['loaded_pretrained_names'])
        target = model.state_dict()
        shared = sorted(set(source) & set(target))
        if any(source[k].shape != target[k].shape for k in shared):
            raise ValueError('Unexpected shared parameter shape change')
        added = sorted(set(target)-set(source))
        if any(not k.startswith(('swin_unet.encoder_h_exchanges.', 'swin_unet.h_surface_residual.')) for k in added):
            raise ValueError(f'Unexpected H256 additions: {added}')
        for key in shared:
            target[key] = source[key]
        model.load_state_dict(target, strict=True)
        if any(not torch.equal(model.state_dict()[k], source[k]) for k in shared):
            raise RuntimeError('H256 retained initial parameters are not identical to a668711')
        digest = hashlib.sha256()
        for key in shared:
            digest.update(key.encode()); digest.update(source[key].cpu().contiguous().numpy().tobytes())
        manifest = dict(base_commit=EXPERIMENT_BASE, original_h0_commit=BASE_COMMIT, seed=seed,
            source=str(base_checkpoint) if base_checkpoint else 'fresh a668711 initialized by its original H0/ImageNet policy',
            shared_tensors=shared, deleted_tensors=sorted(set(source)-set(target)), new_tensors=added,
            shared_tensors_equal=True, shared_tensor_sha256=digest.hexdigest(),
            stem_stride_semantics_change='down2 Conv16->32 weight retained; stride2->1; H native128->256',
            baseline_initialization=original_manifest, original_parameters=sum(p.numel() for p in reference.parameters()),
            new_parameters=sum(p.numel() for p in model.parameters()), optimizer_migrated=False)
        return loaded_names & set(dict(model.named_parameters())), manifest
    if base_checkpoint:
        raise ValueError('base_checkpoint is only valid for explicit a668711-to-H256 weight migration')
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
    if checkpoint.get('architecture_id') not in (ARCHITECTURE_ID, H256_ID) or checkpoint.get('base_h0_commit') != BASE_COMMIT:
        raise ValueError('This entry requires the new Stage2-skeleton/Stage3-diffusion checkpoint')
    arc = checkpoint['architecture_config']
    if arc['architecture_id'] != checkpoint['architecture_id']:
        raise ValueError('Checkpoint architecture ID fields disagree')
    if arc['architecture_id'] == H256_ID and checkpoint.get('base_experiment_commit') != EXPERIMENT_BASE:
        raise ValueError('H256 checkpoint must record the exact a668711 baseline')
    model = build_model(arc)
    if arc['architecture_id'] == H256_ID and checkpoint.get('model_config') != model.config.dump():
        raise ValueError('H256 checkpoint complete Swin configuration differs from this fixed protocol')
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
