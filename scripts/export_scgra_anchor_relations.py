"""Capture same-anchor SCGRA relation terms and render a fixed query on the test image."""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics import scgra_decoder_mass as mass
from mass_protocol import sliding_positions
from scripts.export_scgra_decoder_mass import dump, native_rgb, save_png

MODES = ('base', 'conn', 'pos', 'full')
TITLES = ('Base attention', 'Base + connectivity', 'Base + spatial', 'Full relations')


def capture(model, tile):
    module = model.swin_unet.global_topology
    with torch.inference_mode():
        reference = model(tile)[0].clone()
        module.capture_relation_terms = True
        module.relation_term_calls = 0
        try:
            observed = model(tile)[0]
            if not torch.equal(reference, observed):
                raise RuntimeError('Relation capture changed the normal surface prediction')
            if module.relation_term_calls != 1:
                raise RuntimeError('Expected exactly one original decoder SCGRA call per window')
            result = module.last_relation_terms
        finally:
            del module.capture_relation_terms
            del module.relation_term_calls
            if hasattr(module, 'last_relation_terms'):
                del module.last_relation_terms
    return {key: (value[0] if value.ndim else value).numpy() for key, value in result.items()}


def check_weights(data):
    valid = data['anchor_valid']
    if not valid.any():
        raise ValueError('The chosen window has no valid anchors')
    for mode in MODES:
        attention = data[f'attention_{mode}'][:, valid]
        if not np.isfinite(attention).all() or not np.allclose(attention.sum(-1), 1, atol=2e-6):
            raise ValueError(f'Invalid attention in {mode}')
        if np.any(attention[..., ~valid] != 0):
            raise ValueError(f'Invalid keys must be masked in {mode}')


def write_query_table(path, data, query, top, left):
    anchors = data['anchor_yx']
    valid_ids = np.flatnonzero(data['anchor_valid'])
    fields = ('head', 'query_id', 'key_id', 'native_y', 'native_x', 'distance_px',
              'base_qk_logit', 'connectivity_bias', 'spatial_bias',
              *[f'weight_{mode}' for mode in MODES])
    with path.open('w', newline='', encoding='utf-8') as stream:
        table = csv.DictWriter(stream, fieldnames=fields)
        table.writeheader()
        for head in range(data['attention_full'].shape[0]):
            for key in valid_ids:
                y, x = anchors[key]
                table.writerow(dict(head=head+1, query_id=query+1, key_id=int(key+1),
                    native_y=float(y+.5+top), native_x=float(x+.5+left),
                    distance_px=float(np.linalg.norm(anchors[key]-anchors[query])),
                    base_qk_logit=float(data['base_qk_logits'][head,query,key]),
                    connectivity_bias=float(data['connectivity_bias_effective'][head,query,key]),
                    spatial_bias=float(data['spatial_bias_effective'][head,query,key]),
                    **{f'weight_{mode}':float(data[f'attention_{mode}'][head,query,key])
                       for mode in MODES}))


def figures(directory, data, rgb, query, image_id, top, left, head):
    from scripts.plot_scgra_results import save
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    valid_ids = np.flatnonzero(data['anchor_valid'])
    anchors = data['anchor_yx'][valid_ids]
    q = int(np.where(valid_ids == query)[0][0])
    matrices = []
    for mode in MODES:
        weights = data['attention_'+mode][:,valid_ids][:,:,valid_ids]
        matrices.append(weights.mean(axis=0) if head == 0 else weights[head-1])
    head_label = 'Mean of all heads' if head == 0 else f'Head {head}'
    header = f'{image_id} | Window ({top}, {left}) | {head_label}'
    diff_max = max(float(np.max(np.abs(m - matrices[0]))) for m in matrices)
    color_limit = diff_max or 1e-12
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.95))
    fig.subplots_adjust(left=.075, right=.89, top=.84, bottom=.135, wspace=.21, hspace=.40)
    for j, (matrix, title) in enumerate(zip(matrices,TITLES)):
        for i, ax in enumerate(axes[:,j]):
            display = matrix if i == 0 else matrix - matrices[0]
            cm = ax.imshow(display, cmap='cividis' if i==0 else 'RdBu_r',
                           vmin=0 if i==0 else -color_limit, vmax=1 if i==0 else color_limit,
                           interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=7)
                absolute = cm
            else:
                difference = cm
        if not j:
            axes[0,j].set_ylabel('Query anchor')
            axes[1,j].set_ylabel('Change from base')
    for artist, ax, label in ((absolute,axes[0,-1],'Weight'),(difference,axes[1,-1],'Delta')):
        box = ax.get_position()
        cax = fig.add_axes([.908,box.y0,.014,box.height])
        cb = fig.colorbar(artist, cax=cax)
        cb.set_label(label, fontsize=6, labelpad=2)
        cb.ax.tick_params(labelsize=5.5, length=1, pad=1)
    fig.text(.075,.955,'Structure-conditioned anchor relations',weight='bold',fontsize=9)
    fig.text(.075,.905,header,fontsize=6.5)
    fig.text(.075,.045,'Same anchor order in every panel. Columns are keys. The lower row shares a signed difference scale.',
             fontsize=6)
    save(fig,axes,directory,'relation_matrices')

    fig, axes = plt.subplots(1,3,figsize=(7.2,3.12))
    fig.subplots_adjust(left=.045,right=.89,top=.80,bottom=.145,wspace=.15)
    norm = Normalize(0,1)
    for i, ax in enumerate(axes):
        ax.imshow(rgb)
        ax.scatter(anchors[:,1],anchors[:,0],s=12,facecolors='none',edgecolors='white',linewidths=.55)
        if i:
            weights = matrices[0 if i == 1 else 3][q]
            selected = [k for k in np.argsort(-weights,kind='stable') if k != q and weights[k] > 0][:5]
            for k in selected:
                ax.plot(anchors[[q,k],1],anchors[[q,k],0],color='#54d5ff',lw=.4+2.2*weights[k],
                        alpha=.22+.78*weights[k])
            mappable = ax.scatter(anchors[:,1],anchors[:,0],c=weights,s=15,cmap='cividis',norm=norm,
                                  edgecolors='#eeeeee',linewidths=.35)
            ax.text(.5,-.05,f'Self weight {weights[q]:.4f}',transform=ax.transAxes,ha='center',va='top',fontsize=6)
        ax.scatter(anchors[q,1],anchors[q,0],s=31,c='#ff9d27',marker='D',
                   edgecolors='#21252a',linewidths=.65,zorder=4)
        ax.set(xlim=(-.5,511.5),ylim=(511.5,-.5))
        ax.set_axis_off()
        ax.set_title(('Anchor locations','Base relations','Full relations')[i],fontsize=7)
    box = axes[-1].get_position()
    bar = fig.add_axes([.91,box.y0,.014,box.height])
    cb = fig.colorbar(mappable,cax=bar)
    cb.set_label('Attention',fontsize=6,labelpad=2)
    cb.ax.tick_params(labelsize=5.5,length=1,pad=1)
    fig.text(.045,.955,f'Spatial attention | Query anchor {query+1}',weight='bold',fontsize=9)
    fig.text(.045,.895,header,fontsize=6.5)
    fig.text(.045,.044,'Orange diamond: query. Lines mark up to five positive non-self weights, ranked by attention.',
             fontsize=6)
    save(fig,axes,directory,'query_on_image')
    return diff_max


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--root_path', required=True, type=Path)
    parser.add_argument('--from_export', required=True, type=Path, help='The completed test49 m1_anchors directory')
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cases', nargs='+', help='Default: same three exported test examples')
    parser.add_argument('--top', type=int, default=512)
    parser.add_argument('--left', type=int, default=512)
    parser.add_argument('--query_index', type=int, help='1-based anchor slot, default nearest valid anchor to tile center')
    parser.add_argument('--head', type=int, default=0, help='1-based head for figures, 0 draws their average; NPZ stores all heads')
    parser.add_argument('--no_figures', action='store_true', help='Export the NPZ, CSV and RGB without Matplotlib')
    args = parser.parse_args()
    metadata_path = args.from_export/'metadata.json'
    meta = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
    checkpoint_sha = mass.file_digest(args.checkpoint)
    positions = sliding_positions(1500,512,256)
    completion = args.from_export/'export_complete.json'
    completed = json.loads(completion.read_text(encoding='utf-8-sig')) if completion.is_file() else {}
    if (not completed.get('complete') or completed.get('images') != 49 or
            completed.get('metadata_sha256') != mass.file_digest(metadata_path) or
            meta['checkpoint_sha256'] != checkpoint_sha or
            meta['split'] != 'test' or len(set(meta['evaluated_case_ids'])) != 49 or
            args.top not in positions or args.left not in positions):
        parser.error('Require the original completed test49 run, its exact checkpoint and an original window start')
    cases = args.cases or meta['export_case_ids']
    if not cases or not set(cases) <= set(meta['evaluated_case_ids']) or len(cases) != len(set(cases)):
        parser.error('Specify unique cases from the original test49 run')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; check CUDA_VISIBLE_DEVICES or use --device cpu')
    data = mass.dataset(args.root_path,'test')
    index = {Path(name).stem:i for i,name in enumerate(data.image_files)}
    if set(index) != set(meta['evaluated_case_ids']):
        raise ValueError('Test image identities differ from the original M1 export')
    model, saved, epoch = mass.load_model(args.checkpoint, device)
    if saved.get('freeze_scgra_off'):
        parser.error('Use the original complete decoder checkpoint')
    state = mass.tensor_digests(model)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for case in cases:
        directory = output/case
        directory.mkdir(exist_ok=True)
        sample_index = index[case]
        archived_case = args.from_export/'partial'/f'{case}.json'
        if archived_case.is_file():
            original = json.loads(archived_case.read_text(encoding='utf-8-sig'))['inputs']
            for key, folder, name in (
                ('image_sha256', data.image_dir, data.image_files[sample_index]),
                ('gt_sha256', data.mask_dir, data.label_files[sample_index]),
            ):
                if mass.file_digest(Path(folder)/name) != original[key]:
                    raise ValueError(f'Test input changed since the M1 export: {case}, {key}')
        _, full, gt = mass.read_sample(data,sample_index)
        region = (slice(args.top,args.top+512),slice(args.left,args.left+512))
        tile = full[:,region[0],region[1]]
        arrays = capture(model, tile[None].to(device))
        check_weights(arrays)
        valid_ids = np.flatnonzero(arrays['anchor_valid'])
        query = (args.query_index-1 if args.query_index is not None else
                 int(valid_ids[np.argmin(np.sum((arrays['anchor_yx'][valid_ids]-255.5)**2,axis=1))]))
        if query not in valid_ids or not 0 <= args.head <= arrays['attention_full'].shape[0]:
            parser.error('Chosen query or attention head is outside the valid slots')
        rgb = native_rgb(tile)
        save_png(directory/'rgb.png', cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        save_png(directory/'gt.png', gt[region].astype(np.uint8)*255)
        arrays['anchor_native_yx'] = arrays['anchor_yx'] + np.array([args.top+.5,args.left+.5])
        arrays['query_index_zero_based'] = np.array(query)
        np.savez_compressed(directory/'relation_terms.npz',**arrays)
        write_query_table(directory/'query_weights.csv', arrays, query, args.top, args.left)
        change = figures(directory,arrays,rgb,query,case,args.top,args.left,args.head) if not args.no_figures else None
        previous = args.from_export/'cases'/case/'windows'/f'y{args.top:04d}_x{args.left:04d}.npz'
        matches_previous = None
        if previous.is_file():
            with np.load(previous,allow_pickle=False) as old:
                matches_previous = bool(np.array_equal(old['selected_coords'],arrays['anchor_yx'])
                                     and np.array_equal(old['selected_valid'],arrays['anchor_valid'])
                                     and np.allclose(old['node_attention'],arrays['attention_full'],rtol=1e-5,atol=1e-6))
            if not matches_previous:
                print(f'[CHECK] {case}: the archived M1 full trace differs; see summary.json',flush=True)
        summary.append(dict(case_id=case,query_index=query+1,valid_anchors=len(valid_ids),
            max_delta_for_rendered_head=change, matches_previous_m1_full_trace=matches_previous,
            max_absolute_weight_change={mode:float(np.max(np.abs(
                arrays[f'attention_{mode}'][:,valid_ids]-arrays['attention_base'][:,valid_ids]))) for mode in MODES}))
        print(f'[SCGRA] {case} | query {query+1} | {directory}',flush=True)
    if state != mass.tensor_digests(model):
        raise RuntimeError('SCGRA tracing changed model state')
    dump(output/'summary.json',dict(checkpoint_sha256=checkpoint_sha, checkpoint_epoch=epoch, split='test',
        window_top_left=[args.top,args.left], tile_size=512, head_for_figures=args.head, threshold_scope='no road binarization used',
        query_rule='fixed 1-based slot' if args.query_index is not None else 'valid anchor nearest the window center in Euclidean distance',
        modes=dict(base='QK / sqrt(head dim)',conn='base + effective connectivity bias',
                   pos='base + effective spatial bias',full='actual model attention with both biases'),
        forward_and_model_state_unchanged=True,cases=summary))


if __name__ == '__main__':
    main()
