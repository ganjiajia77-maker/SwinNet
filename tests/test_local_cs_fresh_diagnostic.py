import json
from pathlib import Path
import pytest
import torch
from scripts.evaluate_local_cs import check_comparison, relation_changes


def test_fresh_requires_same_full_initialization(tmp_path):
    paths = {}
    for name in ('initial', 'c_guided'):
        directory = tmp_path/name; directory.mkdir()
        paths[name] = directory/'best.pth'
        (directory/'initial_state.json').write_text(json.dumps(dict(
            seed=1234, pretrain_sha256='pretrain', native_relations=True,
            tensors={'encoder.weight': {'sha256':'identical','shape':[1]}})))
    report = check_comparison(paths, 'independent')
    assert report['reference_role'] == 'original_B'
    assert report['evidence']['c_guided']['identical_initialization']
    path = paths['c_guided'].parent/'initial_state.json'
    data = json.loads(path.read_text()); data['tensors']['encoder.weight']['sha256'] = 'changed'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='initialization'):
        check_comparison(paths, 'independent')


def test_multiple_thresholds_use_separate_fixed_reference_groups():
    good = torch.ones(4, dtype=torch.bool)
    y = torch.tensor([True,True,False,False])
    hard = torch.tensor([False,False,True,False])
    confidence = torch.full((4,), .9)
    initial = torch.tensor([.3,.8,.1,.7])
    current = torch.tensor([.6,.1,.4,.9])
    low = relation_changes(good,y,hard,confidence,initial,current,.2,.5)
    high = relation_changes(good,y,hard,confidence,initial,current,.46,.5)
    assert low['initial_c_supported_missed_positive'] == low['recovered_positive'] == 0
    assert high['initial_c_supported_missed_positive'] == high['recovered_positive'] == 1
    assert low['lost_positive'] == high['lost_positive'] == 1
    assert low['new_hard_negative_connection'] == 1
    assert high['new_hard_negative_connection'] == 0
    reference = relation_changes(good,y,hard,confidence,initial,initial,.46,.5)
    assert all(reference[k] == 0 for k in ('recovered_positive','lost_positive','new_negative_connection'))


def test_independent_full_pipeline_writes_separate_threshold_tables(tmp_path, monkeypatch):
    import csv
    import sys
    import numpy as np
    from scripts import evaluate_local_cs as audit
    torch.set_num_threads(2)
    paths = {}
    manifest = dict(seed=1234,pretrain_sha256='same',native_relations=True,tensors={'w':'same'})
    for name in ('initial','c_guided'):
        directory = tmp_path/name; directory.mkdir()
        paths[name] = directory/'best.pth'; paths[name].write_bytes(name.encode())
        (directory/'initial_state.json').write_text(json.dumps(manifest))
    road = torch.zeros(1,1500,1500); road[0,750,200:1300] = 1
    image = road.expand(3,-1,-1).clone()

    class Dataset:
        def __init__(self,*args,**kwargs): pass
        def __len__(self): return 1
        def __getitem__(self,index):
            return dict(image=image,mask=road,skeleton=road,case_name='toy')

    class Model(torch.nn.Module):
        def __init__(self,candidate): super().__init__(); self.candidate=candidate
        def forward(self,x):
            probability = torch.where(x[:,:1]>.5, .6 if self.candidate else .3, .01)
            z = torch.logit(probability)
            return (z,None,None,None,[dict(stage=3,connectivity=torch.full((1,8,128,128),3.))])

    def load(path,device,native):
        return Model(path == paths['c_guided']), dict(local_cs_mode='c_guided' if path == paths['c_guided'] else 'off')
    fixed = tmp_path/'gaps.json'
    flat = lambda col: 750*1500+col
    fixed.write_text(json.dumps(dict(count=1,images={'toy':{'gaps':[
        dict(id=77,pixels=[flat(x) for x in range(740,750)],anchors=[[flat(735)],[flat(755)]])
    ]}})))
    out=tmp_path/'diagnostic'
    monkeypatch.setattr(audit,'load_model',load)
    monkeypatch.setattr(audit,'RoadSkeletonDataset',Dataset)
    monkeypatch.setattr(audit,'sliding_positions',lambda *args:[0,512,988])
    monkeypatch.setattr(audit,'bridge_proxy',lambda *args:0)
    monkeypatch.setattr(audit,'skeletonize',lambda x:x)
    monkeypatch.setattr(sys,'argv',['audit','--comparison_kind','independent',
        '--reference',str(paths['initial']),'--run','c_guided='+str(paths['c_guided']),
        '--root_path',str(tmp_path),'--fixed_gaps',str(fixed),'--output_dir',str(out),
        '--device','cpu','--diagnostic_thresholds','.2','.46'])
    audit.main()
    with (out/'relations.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==4
    candidate = [r for r in rows if r['model']=='c_guided']
    assert int(candidate[0]['recovered_positive'])==0
    assert int(candidate[1]['recovered_positive'])>0
    with (out/'fixed_gap_summary.csv').open() as f: gaps=list(csv.DictReader(f))
    selected=[r for r in gaps if r['model']=='c_guided']
    assert [int(r['newly_closed']) for r in selected]==[0,1]
    protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['reference_role']=='original_B'
    assert len(protocol['diagnostic_thresholds'])==2
    assert (out/'fixed_gap_changes.csv').is_file()
