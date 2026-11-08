"""Read initialization, target verification and training activation logs."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir',type=Path,required=True)
    a=p.parse_args()
    for directory in (Path(str(a.run_dir)+'_smoke'),a.run_dir):
        print('\n========== '+str(directory)+' ==========')
        for name in ('original_b_initialization_check.json','native_smoke.json'):
            path=directory/name
            if path.exists():
                print(name+'\n'+json.dumps(json.loads(path.read_text()),indent=2,ensure_ascii=False))
        for name in ('native_target_verification.jsonl','h_reassembly_training.jsonl'):
            path=directory/name
            if not path.exists():continue
            # Bounded memory even for a complete 80-epoch log.
            from collections import deque
            with path.open() as f:lines=deque((line for line in f if line.strip()),maxlen=5)
            print('\nLatest '+name)
            for line in lines:print(json.dumps(json.loads(line),ensure_ascii=False))
        log=directory/'training_log.txt'
        if log.exists():
            from collections import deque
            with log.open(encoding='utf-8',errors='replace') as f:lines=deque(f,maxlen=15)
            print('\nTraining progress\n'+''.join(lines))


if __name__=='__main__':main()
