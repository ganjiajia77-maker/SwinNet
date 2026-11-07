"""No pandas required; verify paired inputs and show short-run cost/gradients."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--pair', required=True)
    a = p.parse_args()
    rows, hashes, inputs, initial_states = [], [], [], []
    for mode in ('off', 'surface_only', 'c_guided'):
        directory = a.results / (a.pair+'_'+mode)
        if not (directory/'local_cs_training.jsonl').is_file():
            continue
        records = [json.loads(x) for x in (directory/'local_cs_training.jsonl').read_text().splitlines()]
        initial = json.loads((directory/'local_cs_initial.json').read_text())
        hashes.append(initial['sha256'])
        initial_states.append(json.loads((directory/'initial_state.json').read_text()))
        inputs.append([json.loads(f.read_text()) for f in sorted(directory.glob('input_epoch_*.json'))])
        audited = [r for r in records if 'gradient_ratio' in r]
        timings = [r['ms_per_batch'] for r in records if 'gradient_ratio' not in r and r['batch'] > 5]
        rows.append(dict(mode=mode, batches=len(records), ms_per_batch=mean(timings) if timings else 0,
            raw_loss=mean(r.get('raw', 0) for r in records),
            weighted_loss=mean(r.get('weighted', 0) for r in records),
            gradient_ratio_max=max((r['gradient_ratio'] for r in audited), default=0),
            positive=sum(r.get('positive', 0) for r in records),
            negative=sum(r.get('negative', 0) for r in records),
            positive_empty_images=sum(r.get('positive_empty_images', 0) for r in records),
            negative_empty_images=sum(r.get('negative_empty_images', 0) for r in records)))
    if not rows:
        raise RuntimeError('No completed batch records')
    if len(set(hashes)) != 1 or any(x != initial_states[0] for x in initial_states):
        raise RuntimeError('Initial weights differ')
    if any(x != inputs[0] for x in inputs):
        raise RuntimeError('Image order, augmented RGB or completed budget differs')
    base = next((r['ms_per_batch'] for r in rows if r['mode'] == 'off'), None)
    for r in rows:
        r['time_ratio_to_off'] = r['ms_per_batch']/base if base else None
    output = a.results/(a.pair+'_summary.csv')
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(shared_initial_checkpoint_sha256=hashes[0], paired_inputs_equal=True, runs=rows), indent=2))
    print(output)


if __name__ == '__main__':
    main()
