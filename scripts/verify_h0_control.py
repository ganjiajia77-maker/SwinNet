"""Print deterministic state/output digests for original and disabled-exchange H0."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def digest_state(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    root = args.source_root.resolve()
    sys.path.insert(0, str(root))
    from config import get_config
    from networks.vision_transformer import SwinUnet

    config_args = SimpleNamespace(
        cfg=str(root / 'configs/swin_tiny_patch4_window7_224_lite.yaml'),
        opts=None, batch_size=1, img_size=512, zip=False, cache_mode='',
        resume='', accumulation_steps=1, use_checkpoint=False,
        amp_opt_level='', tag='', eval=False, throughput=False,
    )
    torch.manual_seed(2026)
    model = SwinUnet(
        get_config(config_args), img_size=512, num_classes=1, return_skeleton=True,
        structure_profile='stage23_boundary_0626',
        enable_highres_structure_stream=True, highres_structure_channels=64,
        highres_structure_fuse_stages='stage23',
        highres_structure_fusion_mode='stage23',
        enable_global_topology=True, global_topology_max_nodes=32,
        global_topology_heads=4, global_topology_alpha_max=0.05,
        enable_h3_surface_fusion=True, remove_stage2_pre_topology_source=True,
        stage_skeleton_mode='direct',
    )
    state_digest = digest_state(model.state_dict())
    model = model.to(args.device).eval()
    image = torch.randn(1, 3, 512, 512, device=args.device)
    with torch.no_grad():
        logits = model(image)[0]
    output_digest = hashlib.sha256(logits.detach().cpu().numpy().tobytes()).hexdigest()
    print(json.dumps({'source': str(root), 'parameters': sum(p.numel() for p in model.parameters()),
                      'state_sha256': state_digest, 'output_sha256': output_digest,
                      'output_sum': float(logits.sum().item())}), flush=True)


if __name__ == '__main__':
    main()
