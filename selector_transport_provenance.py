"""Read server artifacts before creating the actual-transport experiment."""
import json
from pathlib import Path

import torch

import run_reliable_v9_mass as v9
from networks.v9_source_selector import VERSION as SELECTOR_VERSION
from networks.v9_transport_benefit import BENEFIT_VERSION

SELECTION_RULE='IoU + 0.15*clDice + 0.10*APLS_GT_to_Pred + 0.05*APLS_bidirectional - 0.001*short_pred_components - 0.01*background_bridge_proxy/images'
CONTROL_SELECTION='original V9 weighted score at .60 for checkpoints; weighted score over val .35:.01:.75 for thresholds'


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def check_code_hashes(config):
    root=Path(__file__).resolve().parent
    for name,expected in config.get('code_sha256',{}).items():
        if not (root/name).is_file() or v9.digest(root/name)!=expected:
            raise ValueError(f'The validated artifact used different code: {name}')


def same_state(left,right):
    return set(left)==set(right) and all(torch.equal(left[key],right[key]) for key in left)


def inspect_selector(run,results,original_run=None):
    run=Path(run).resolve(); config=read_json(run/'experiment.json')
    report=read_json(run/'relation_best/relations.json')
    if not report.get('screening',{}).get('passed'): raise ValueError('Selector fixed-relation validation did not pass')
    state=torch.load(run/'relation/best.pth',map_location='cpu',weights_only=False)
    if state.get('version')!=SELECTOR_VERSION or state.get('phase')!='relation' or state.get('config')!=config:
        raise ValueError('Relation checkpoint does not correspond to this validated run/config')
    if state.get('identity')!=config.get('identity') or state.get('source_sha256')!=config.get('source_sha256'):
        raise ValueError('Selector source identity/hash is inconsistent')
    if abs(float(state['score'])-float(report['selection_score']))>1e-8:
        raise ValueError('Fixed relation report score does not correspond to the saved best checkpoint')
    check_code_hashes(config)
    for name in ('base','q3'):
        dependency=state['identity'][name]; dependency_path=Path(dependency['path'])
        if not dependency_path.is_file() or v9.digest(dependency_path)!=dependency['sha256']:
            raise ValueError(f'Frozen {name} is missing or differs from the validated selector identity')
    if original_run is not None: candidates=[Path(original_run)/'stage_D/best.pth']
    else: candidates=list(Path(results).glob('*/stage_D/best.pth'))
    matches=[path for path in candidates if path.is_file() and v9.digest(path)==state['source_sha256']]
    if not matches: raise FileNotFoundError('Cannot find the exact original V9 by selector source SHA256')
    verified=[]
    for path in matches:
        selection_path=path.parent.parent/'direct_val_selection.json'
        if not selection_path.is_file(): continue
        source=torch.load(path,map_location='cpu',weights_only=False); selection=read_json(selection_path)
        if source.get('version')!=v9.VERSION or source.get('stage')!='D' or source.get('identity')!=state['identity']: continue
        if selection.get('identity')!=source['identity'] or selection.get('selected_on')!='val': continue
        if Path(selection['checkpoint']).resolve()!=path.resolve(): continue
        for name in v9.MODULE_NAMES:
            if not same_state(state['modules'][name],source['modules'][name]):
                raise ValueError(f'Selector pretraining changed original V9 module {name}')
        for name in ('gate_relation','state_gate'):
            if not same_state(state[name],source[name]): raise ValueError(f'Selector pretraining changed {name}')
        verified.append((path,source,selection))
    if not verified: raise ValueError('No matching original V9 checkpoint and validation selection metadata')
    path,source,selection=verified[0]
    if float(source['training_config']['reference_threshold'])!=.45 or selection['report']['paired']['eligible_p0_gaps']!=206:
        raise ValueError('Expected the original fixed .45 /206-gap reference')
    return {'run':run,'config':config,'checkpoint':run/'relation/best.pth','state':state,
            'original_path':path,'original':source,'selection':selection,'report':report}


def inspect_control(run,selector,epochs):
    run=Path(run)
    if not (run/'config.json').is_file(): run=run/'control_3step'
    config=read_json(run/'config.json'); state=torch.load(run/'best.pth',map_location='cpu',weights_only=False)
    last=torch.load(run/'last.pth',map_location='cpu',weights_only=False); completed=read_json(run/'training_complete.json')
    if config.get('variant')!='control' or config.get('steps')!=3 or config.get('epochs')!=epochs:
        raise ValueError('Not a same-budget three-step B_control continuation')
    if config.get('identity')!=selector['state']['identity'] or config.get('original_sha256')!=selector['state']['source_sha256']:
        raise ValueError('B_control has a different original V9/BASE/Q3 identity')
    original=selector['original']; stage_config=selector['config']; training=original['training_config']
    for key,expected in (('root_path',stage_config['root_path']),('seed',stage_config['seed']),('batch_size',2),
                         ('num_workers',stage_config['num_workers']),('learning_rates',original['learning_rates']),
                         ('selection',CONTROL_SELECTION)):
        if config.get(key)!=expected: raise ValueError(f'B_control differs in {key}')
    if config.get('val_interval',0)<1: raise ValueError('Missing B_control checkpoint selection cadence')
    if not all(token in config.get('protocol','') for token in ('FP32','no EMA','accumulation1')):
        raise ValueError('B_control precision/EMA/accumulation protocol is not the expected one')
    check_code_hashes(config)
    for payload in (state,last):
        if payload.get('version')!=BENEFIT_VERSION or payload.get('benefit_config')!=config or payload.get('stage')!='D' or payload.get('training_config')!=training:
            raise ValueError('B_control checkpoint configuration does not match original loss/training protocol')
    if last['epoch']!=epochs or completed.get('epochs')!=epochs or completed.get('variant')!='control':
        raise ValueError('B_control continuation is not complete')
    groups=last.get('optimizer',{}).get('param_groups',[])
    if {g.get('name') for g in groups}!=set(v9.STAGE_MODULES['D']):
        raise ValueError('B_control did not train the original V9 Stage-D module range')
    for group in groups:
        if group['lr']!=original['learning_rates'][group['name']] or group.get('weight_decay')!=.01:
            raise ValueError('B_control optimizer/lr/weight-decay differs')
    records={item['epoch']:item for line in (run/'epochs.jsonl').read_text().splitlines() if line.strip() for item in [json.loads(line)]}
    if set(records)!=set(range(1,epochs+1)): raise ValueError('B_control epoch/data records are incomplete')
    for row in records.values():
        loss=row['losses']
        if abs(loss['total']-(loss['seg']+.2*loss['cont']))>2e-5:
            raise ValueError('B_control loss contains terms beyond Lseg+.2Lcont')
        if row.get('gradient_comparison_first_matched_batch',{}):
            if row['gradient_comparison_first_matched_batch'].get('gate_loss_applied_to_optimizer') is not False:
                raise ValueError('Cannot prove that L_G was absent from B_control optimizer loss')
    validation={epoch:row['val_score_at_0_60'] for epoch,row in records.items() if 'val_score_at_0_60' in row}
    expected_epochs={epoch for epoch in records if epoch%config['val_interval']==0 or epoch==epochs}
    if set(validation)!=expected_epochs or state['epoch'] not in validation or abs(float(state['score'])-max(validation.values()))>1e-8:
        raise ValueError('B_control best checkpoint does not follow its recorded val@0.60 selection rule')
    val_report=read_json(run/f'val_epoch_{state["epoch"]:03d}.json')
    if abs(float(val_report['selection_score'])-float(state['score']))>1e-8:
        raise ValueError('B_control best validation report/score mismatch')
    return {'run':run,'config':config,'state':state,'last':last,'records':records,'checkpoint':run/'best.pth','val_report':val_report}


def find_control(results,selector,epochs,explicit=None):
    if explicit is not None: return inspect_control(explicit,selector,epochs)
    paths=set(Path(results).glob('*/control_3step'))
    marker=Path(results)/'latest_v9_gate_benefit_run.txt'
    if marker.is_file(): paths.add(Path(marker.read_text().strip())/'control_3step')
    matches={}; rejected={}
    for path in sorted(paths,key=str):
        try:
            value=inspect_control(path,selector,epochs)
            matches.setdefault(v9.digest(value['checkpoint']),value)
        except (ValueError,KeyError,OSError,TypeError) as error: rejected[str(path)]=str(error)
    if len(matches)!=1:
        raise ValueError(json.dumps({'error':'Need one compatible existing B_control; no automatic retraining',
            'compatible':[str(value['run']) for value in matches.values()],'rejected':rejected},indent=2))
    return next(iter(matches.values()))


def find_reference(original_run,explicit=None):
    if explicit is not None: paths=[Path(explicit)]
    else: paths=sorted(Path(original_run).rglob('reference_set.json'),key=str)
    found={}
    for path in paths:
        value=read_json(path)
        if value.get('eligible_reference_gaps')==206 and value.get('images')==14 and value.get('p0_reference_threshold')==.45:
            found.setdefault(value['sha256'],path)
    if len(found)!=1: raise ValueError('Need one unambiguous existing .45/206-gap original diagnostic reference_set.json')
    return next(iter(found.values()))
