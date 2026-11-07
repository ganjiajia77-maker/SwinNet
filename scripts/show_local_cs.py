"""Show all short-run tables without pandas."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True);a=p.parse_args()
    base=a.prefix
    tables=[Path(str(base)+'_summary.csv')]
    pointer=Path(str(base)+'_eval_latest.txt')
    if pointer.exists():
        out=Path(pointer.read_text().strip())
        tables += [out/name for name in ('comparison.csv','relations.csv','fixed_gap_changes.csv','topology/val_comparison.csv')]
    for path in tables:
        print('\n========== '+str(path)+' ==========')
        if not path.exists():print('Not generated yet');continue
        with path.open(newline='',encoding='utf-8-sig') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
        if not fields:continue
        cells=[[str(row.get(k,'')) for k in fields] for row in rows]
        widths=[max(len(k),max((len(row[i]) for row in cells),default=0)) for i,k in enumerate(fields)]
        print(' | '.join(k.ljust(w) for k,w in zip(fields,widths)))
        print('-+-'.join('-'*w for w in widths))
        for row in cells:print(' | '.join(v.ljust(w) for v,w in zip(row,widths)))
    for mode in ('off','surface_only','c_guided'):
        f=Path(str(base)+'_'+mode)/'local_cs_training.jsonl'
        if f.exists():
            lines=f.read_text().splitlines()
            print('\nLatest batch: '+mode)
            print(json.dumps(json.loads(lines[-1]),indent=2))


if __name__=='__main__':main()
