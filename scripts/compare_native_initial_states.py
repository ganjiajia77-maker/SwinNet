"""Verify equal checkpoint initialization for shared H0 tensors."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('native',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    a=json.loads(args.baseline.read_text());b=json.loads(args.native.read_text())
    if a['native_relations'] or not b['native_relations']:raise ValueError('Expected H0 baseline and native candidate')
    if a['seed']!=b['seed'] or a['pretrain_sha256']!=b['pretrain_sha256']:raise ValueError('Seed or pretraining differs')
    excluded='swin_unet.decoder_structure_blocks.3.skeleton_head.'
    common=sorted((set(a['tensors'])&set(b['tensors']))-{name for name in a['tensors'] if name.startswith(excluded)})
    if not common:raise ValueError('No shared tensors')
    required=('swin_unet.patch_embed.','swin_unet.decoder_structure_blocks.2.',
              'swin_unet.decoder_structure_blocks.3.connectivity_head.',
              'swin_unet.global_topology.token_relation_qkv.', 'swin_unet.guided_head.')
    for prefix in required:
        if not any(name.startswith(prefix) for name in common):raise ValueError('Missing shared component: '+prefix)
    mismatches=[name for name in common if a['tensors'][name]!=b['tensors'][name]]
    report={'same_seed':True,'pretrain_sha256':a['pretrain_sha256'],'shared_tensors_checked':len(common),
            'mismatches':mismatches,'required_components':list(required),'excluded_replaced_stage3_skeleton':True,
            'only_baseline':sorted(set(a['tensors'])-set(b['tensors'])),
            'only_native':sorted(set(b['tensors'])-set(a['tensors']))}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    if mismatches:raise ValueError('Shared tensor initialization differs: '+', '.join(mismatches[:10]))
    print(f'[INITIALIZATION MATCH] shared tensors: {len(common)}; report: {args.output}',flush=True)


if __name__=='__main__':main()
