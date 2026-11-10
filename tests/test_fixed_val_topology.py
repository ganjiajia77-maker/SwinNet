import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluate_fixed_val_topology import run


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def setup_case(root):
    protocol = dict(image_size=1500, apls_max_control_nodes=150,
                    topo_seed_density_deg=.002, topo_step_deg=.0002)
    models = [dict(name=name, provenance=dict(sha256='checkpoint'),
                   predictions=dict(val=[dict(path='masks/'+name, threshold=.46)]))
              for name in ('normal', 'c_prior_one', 'residual_off')]
    write(root/'manifest.json', dict(protocol=protocol, models=models))
    write(root/'protocol.json', dict(split='val', threshold=.46, checkpoint_sha256='checkpoint'))
    write(root/'inference_complete.json', dict(complete=True, images=14, checkpoint_sha256='checkpoint'))
    failed = root/'topology'
    failed.mkdir()
    (failed/'val_selection.json').write_text('old selection, preserve it')
    (failed/'val_threshold_scores.csv').write_text('old threshold table, preserve it')


def fake_tool(calls):
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def evaluate(manifest, base, protocol, split, selection, output, progress, workers, resume):
        # Reproduce the real evaluate() guard that caused the server failure.
        if not (output/'run_state.json').exists():
            assert not list(output.iterdir()), 'Selection files must not pollute output before evaluate'
            write(output/'run_state.json', dict(identity='test'))
            assert not resume
        else:
            assert resume
        assert split == 'val' and workers == 4 and progress == 1
        assert selection['selected_on'] == 'val'
        assert selection['criterion'] == 'fixed_shared_diagnostic_threshold'
        assert {v['threshold'] for v in selection['models'].values()} == {.46}
        assert protocol['apls_max_control_nodes'] == 150
        assert protocol['topo_seed_density_deg'] == .002
        calls.append(dict(resume=resume, output=output, selection=selection))
        return {'ok': True}

    def forbidden(*args, **kwargs):
        raise AssertionError('Must not select or sweep thresholds')

    return SimpleNamespace(validate_manifest=lambda m: m['protocol'], digest=digest,
        threshold_protocol_hash=digest, threshold_of=lambda s: float(s['threshold']),
        model_identity=lambda m: digest(m['provenance']), write_json=write,
        evaluate=evaluate, select_validation=forbidden)


def test_preserves_failed_directory_uses_empty_output_and_resumes(tmp_path):
    setup_case(tmp_path)
    calls = []
    module = fake_tool(calls)
    result = run(tmp_path, Path('unused.py'), module=module)
    assert result == {'ok': True}
    assert (tmp_path/'topology'/'val_selection.json').read_text() == 'old selection, preserve it'
    assert len(list((tmp_path/'topology').iterdir())) == 2
    assert Path((tmp_path/'topology_result_dir.txt').read_text().strip()) == tmp_path/'topology_fixed_val'
    run(tmp_path, Path('unused.py'), module=module)
    assert [entry['resume'] for entry in calls] == [False, True]


@pytest.mark.parametrize('change', ['threshold', 'checkpoint', 'incomplete'])
def test_rejects_provenance_mismatch_without_running_metrics(tmp_path, change):
    setup_case(tmp_path)
    path = tmp_path/'manifest.json'
    value = json.loads(path.read_text())
    if change == 'threshold':
        value['models'][1]['predictions']['val'][0]['threshold'] = .2
        write(path, value)
    elif change == 'checkpoint':
        value['models'][1]['provenance']['sha256'] = 'different'
        write(path, value)
    else:
        write(tmp_path/'inference_complete.json', dict(complete=False, images=13))
    calls = []
    with pytest.raises(ValueError):
        run(tmp_path, Path('unused.py'), module=fake_tool(calls))
    assert not calls
    assert not (tmp_path/'topology_fixed_val').exists()
