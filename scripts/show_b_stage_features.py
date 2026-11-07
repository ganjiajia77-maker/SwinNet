"""Show diagnostic tables using Python's standard library only."""
import argparse
import csv
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output_dir',type=Path,required=True);a=p.parse_args()
    for name in ('REPORT.md','checks.json','stages.csv','stage_metrics.csv','adjacent_stage_changes.csv'):
        path=a.output_dir/name
        print('\n========== '+name+' ==========')
        if not path.exists():print('Not generated yet');continue
        if path.suffix!='.csv':print(path.read_text(encoding='utf-8'));continue
        with path.open(newline='',encoding='utf-8') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
        if not fields:continue
        values=[]
        for row in rows:
            cells=[]
            for key in fields:
                value=row[key]
                try:value=f'{float(value):.5f}' if '.' in value else value
                except ValueError:pass
                cells.append(value)
            values.append(cells)
        widths=[max(len(key),max((len(row[i]) for row in values),default=0)) for i,key in enumerate(fields)]
        print(' | '.join(key.ljust(w) for key,w in zip(fields,widths)))
        print('-+-'.join('-'*w for w in widths))
        for row in values:print(' | '.join(value.ljust(w) for value,w in zip(row,widths)))
    print('\nGallery:',a.output_dir/'index.html')
    print('Full-resolution probabilities:',a.output_dir/'predictions')
    print('Gap status:',a.output_dir/'gap_status.csv')


if __name__=='__main__':main()
