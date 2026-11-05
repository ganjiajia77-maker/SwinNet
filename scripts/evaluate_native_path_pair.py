"""Compare two native B checkpoints with the pre-existing 198 fixed gaps."""
import os
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_native_h0_pair as evaluator


def main():
    source=os.environ.get('FIXED_GAPS')
    if not source:raise ValueError('Set FIXED_GAPS to the fixed 198-gap manifest')
    gaps=json.loads(Path(source).read_text(encoding='utf-8'))
    if gaps.get('count')!=198:raise ValueError('Expected the shared 198-gap manifest')
    original_load=evaluator.load_model
    def load_native(path,device,_native):return original_load(path,device,True)
    evaluator.load_model=load_native
    evaluator.fixed_reference=lambda cases:gaps
    evaluator.main()

if __name__=='__main__':main()
