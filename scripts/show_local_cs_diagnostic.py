"""Display independent/continued local-CS diagnostic tables without pandas."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--all_gaps', action='store_true')
    a = p.parse_args()
    print('Directory:', a.output_dir)
    protocol = a.output_dir/'protocol.json'
    if protocol.exists():
        print('\nPROTOCOL\n'+json.dumps(json.loads(protocol.read_text()), indent=2, ensure_ascii=False))
    print('\ninitial = the fixed reference checkpoint (original B in independent mode).')
    tables = [
        ('C-supported positive recovery', 'relations.csv',
         ['model','threshold','initial_c_supported_missed_positive','recovered_positive','recovery_rate']),
        ('Connection losses and new wrong contacts', 'relations.csv',
         ['model','threshold','lost_positive','loss_rate','new_negative_connection',
          'new_hard_negative_connection','new_hard_negative_rate']),
        ('Fixed gap summary (local proxies)', 'fixed_gap_summary.csv', None),
        ('Segmentation, same thresholds and own val threshold', 'comparison.csv', None),
    ]
    if a.all_gaps:
        tables.append(('All fixed gap IDs', 'fixed_gap_changes.csv', None))
    for title, filename, fields in tables:
        path = a.output_dir/filename
        print('\n========== '+title+' ==========')
        if not path.exists():
            print('Not generated:', path)
            continue
        with path.open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f); rows = list(reader)
            fields = fields or reader.fieldnames
        if not rows:
            print('No rows')
            continue
        cells = []
        for row in rows:
            values = []
            for field in fields:
                value = row.get(field, '')
                try:
                    if value and ('.' in value or 'e-' in value):
                        value = f'{float(value):.6f}'
                except ValueError:
                    pass
                values.append(value)
            cells.append(values)
        widths = [max(len(k), max(len(r[i]) for r in cells)) for i,k in enumerate(fields)]
        print(' | '.join(k.ljust(w) for k,w in zip(fields,widths)))
        print('-+-'.join('-'*w for w in widths))
        for row in cells:
            print(' | '.join(v.ljust(w) for v,w in zip(row,widths)))


if __name__ == '__main__':
    main()
