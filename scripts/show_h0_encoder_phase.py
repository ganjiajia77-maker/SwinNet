"""Show initialization/configuration/preflight/training evidence without pandas."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run_dir', type=Path, required=True)
args=parser.parse_args()
root=args.run_dir
sampling=Path(str(root)+'_sampling.json')
print('\n========== 跨轮随机裁剪检查 ==========')
if sampling.is_file():
    data=json.loads(sampling.read_text())
    print('通过={} | 每图不同位置轮数={} | worker输出一致={} | persistent_workers={}'.format(
        data['passed'],data['checked_unique_epochs'],data['worker_direct_equal'],data['persistent_workers']))
    print(f"{'epoch':>6} {'image':28} {'top':>6} {'left':>6}")
    for row in data['samples']:
        print(f"{row['epoch_index']+1:6d} {row['image']:28} {row['top']:6d} {row['left']:6d}")
else:print('尚未生成：',sampling)
for title,path in [('参考H0配置',root/'h0_reference_config.json'),
                   ('共享初始化检查',root/'phase_initial_state.json'),
                   ('512/batch2启动检查',Path(str(root)+'_smoke')/'phase_smoke.json')]:
    print('\n========== '+title+' ==========')
    if not path.is_file():
        print('尚未生成：',path);continue
    data=json.loads(path.read_text())
    if path.name=='phase_initial_state.json':
        data={k:v for k,v in data.items() if k!='initial_tensors'}
    elif path.name=='h0_reference_config.json':
        resolved=data['resolved_args']
        data={k:v for k,v in data.items() if k not in ('reference_args','resolved_args','inherited_keys')}
        data['resolved_training']={k:resolved.get(k) for k in ('max_epochs','batch_size','accumulation_steps',
            'seed','amp_dtype','use_ema','img_size','source_patch_size','random_crop_train','random_crops_per_image',
            'disable_persistent_workers','num_workers','sampling_protocol',
            'max_train_batches','pretrained_lr','new_lr','warmup_epochs','val_interval','threshold',
            'stage2_skeleton_weight','stage3_skeleton_weight','stage_connectivity_factor','stage_direction_factor')}
    print(json.dumps(data,ensure_ascii=False,indent=2))
print('\n========== 最近5条编码器H更新与梯度 ==========')
path=root/'encoder_h_phase_training.jsonl'
if path.is_file():
    for line in path.read_text().splitlines()[-5:]:print(line)
else:print('尚未生成')
print('\n========== 最近逐轮裁剪记录 ==========')
path=root/'sampling_epochs.jsonl'
if path.is_file():
    for line in path.read_text().splitlines()[-5:]:print(line)
else:print('尚未生成')
print('\n========== 最近验证记录 ==========')
path=root/'training_log.txt'
if path.is_file():
    for line in [x for x in path.read_text(encoding='utf-8').splitlines() if 'Surface IoU' in x][-5:]:print(line)
else:print('尚未生成')
