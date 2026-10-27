"""CPU tests of actual downstream modules with a synthetic frozen feature source.

The runner defers backbone imports until a full model is loaded, so these tests
need only CPU tensor/image packages and no pretrained weights.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_reliable_v9_mass as runner
spec = importlib.util.spec_from_file_location('v9_repair', ROOT / 'networks/reliable_directional_repair.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)
NAMES = ('visual', 'reliable_head', 'relations', 'proposal_head', 'accept_head', 'dynamics')
helpers = {name: getattr(runner, name) for name in ('StateGate', 'module', 'configure',
    'grad_norm', 'save_checkpoint', 'restore_checkpoint', 'STAGE_MODULES')}


class FakeStage(nn.Module):
    def __init__(self):
        super().__init__()
        wrapper = nn.Module()
        wrapper.block = nn.ModuleList([nn.Conv2d(96, 16, 1)])
        self.structure_branch = nn.ModuleList([wrapper])


class FrozenSource(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.swin_unet = nn.Module()
        self.base.swin_unet.highres_structure_channels = 64
        self.base.swin_unet.decoder_structure_blocks = nn.ModuleDict({'3': FakeStage()})
        self.features = nn.Conv2d(3, 64, 1)
        self.decoder = nn.Conv2d(64, 96, 1)
        self.surface = nn.Conv2d(64, 1, 1)
        nn.init.constant_(self.surface.bias, -2.)
        self.q = nn.Conv2d(64, 1, 1)

    def forward(self, image, capture_diagnostics=False):
        h = self.features(F.avg_pool2d(image, 4))
        d = self.decoder(F.avg_pool2d(h, 2))
        q = self.q(h).sigmoid()
        return {'z0': F.interpolate(self.surface(h), size=image.shape[-2:], mode='bilinear', align_corners=False),
                'q': q, 'highres_h': h, 'stage3_h': d,
                'stage2_s': q, 'stage2_c': q.expand(-1, 8, -1, -1), 'stage2_d': None}


def model():
    result = repair.ReliableDirectionalRepair(FrozenSource(), transport_gate_mode='v7',
                                              gate_relation_mode='full')
    result.state_gate = helpers['StateGate']()
    return result


class TestV9(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        torch.set_num_threads(1)
        self.model = model()
        self.image = torch.randn(2, 3, 64, 64)
        self.mask = torch.zeros(2, 1, 64, 64)
        self.mask[:, :, 30:33, 4:60] = 1.
        self.skeleton = torch.zeros_like(self.mask)
        self.skeleton[:, :, 31, 4:60] = 1.

    def test_zero_init_and_shared_state_inputs(self):
        self.model.eval()
        state = self.model.state_gate
        self.model.state_gate = None
        with torch.no_grad():
            baseline = self.model(self.image, capture_diagnostics=True)
        self.model.state_gate = state
        seen = []
        handle = state.register_forward_pre_hook(lambda unused, args: seen.append(args[0].detach()))
        with torch.no_grad():
            result = self.model(self.image, capture_diagnostics=True)
        handle.remove()
        self.assertEqual(len(seen), 3)
        for key in ('g_step', 'alpha', 'u1', 'u2', 'u3', 'p1'):
            torch.testing.assert_close(result[key], baseline[key], atol=0., rtol=0.)
        for k, value in enumerate(seen):
            expected = torch.cat((result[f'u{k}'], result['m'][:, k] - result[f'u{k}'],
                result['source_available'][:, k], result['transport_rate'][:, k],
                result['b'] - result[f'u{k}'], result['u0'] - result[f'u{k}']), dim=1)
            torch.testing.assert_close(value, expected)

    def test_state_change_reaches_surface_and_preserves_static_predictions(self):
        self.model.eval()
        with torch.no_grad():
            before = self.model(self.image, capture_diagnostics=True)
            self.model.state_gate.net[-1].bias.fill_(1.)
            after = self.model(self.image, capture_diagnostics=True)
        for key in ('r', 'w', 'b', 'a'):
            torch.testing.assert_close(before[key], after[key], atol=0., rtol=0.)
        self.assertGreater(float((before['g_step'] - after['g_step']).abs().max()), 0.)
        self.assertGreater(float((before['p1'] - after['p1']).abs().max()), 0.)

    def test_phase_coverage_freeze_gradients_and_actual_parameter_updates(self):
        for stage in ('B', 'C', 'D'):
            optimizer = helpers['configure'](self.model, stage)
            self.model.train()
            self.assertFalse(self.model.frozen.training)
            before = {name: [p.detach().clone() for p in helpers['module'](self.model, name).parameters()]
                      for name in helpers['STAGE_MODULES'][stage]}
            norms = {}
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                result = self.model(self.image, capture_diagnostics=True)
                loss, terms = repair.dynamic_losses(result, self.mask, self.skeleton, return_live_terms=True)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(p.grad is None for p in self.model.frozen.parameters()))
                norms = {name: helpers['grad_norm'](helpers['module'](self.model, name)) for name in before}
                optimizer.step()
            for name, value in norms.items():
                self.assertGreater(value, 0., (stage, name, norms))
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before[name], helpers['module'](self.model, name).parameters())), (stage, name))

    def test_checkpoint_roundtrip_and_wrong_upstream_rejection(self):
        optimizer = helpers['configure'](self.model, 'D')
        loss, _ = repair.dynamic_losses(self.model(self.image), self.mask, self.skeleton)
        loss.backward(); optimizer.step()
        self.model.eval()
        with torch.no_grad():
            expected = self.model(self.image)['p1']
        identity = {'base': {'path': 'synthetic', 'sha256': 'synthetic'}}
        with tempfile.TemporaryDirectory(prefix='.v9_checkpoint_test_', dir=ROOT) as tmp:
            path = Path(tmp) / 'last.pth'
            helpers['save_checkpoint'](path, self.model, identity, 'D', 2, .5, optimizer)
            restored = model()
            restored.frozen.load_state_dict(self.model.frozen.state_dict(), strict=True)
            payload = helpers['restore_checkpoint'](restored, path, identity, 'D')
            new_optimizer = helpers['configure'](restored, 'D')
            new_optimizer.load_state_dict(payload['optimizer'])
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(expected, restored(self.image)['p1'], atol=0., rtol=0.)
            with self.assertRaises(ValueError):
                helpers['restore_checkpoint'](restored, path, {'base': 'different'}, 'D')

    def test_frozen_feature_override_matches_live_source(self):
        self.model.eval()
        with torch.no_grad():
            frozen = self.model.frozen(self.image, capture_diagnostics=True)
            live = self.model(self.image)['p1']
            cached = self.model(None, frozen_override=frozen)['p1']
        torch.testing.assert_close(live, cached, atol=0., rtol=0.)

    def test_real_batch_smoke_restores_weights(self):
        before = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        source = {'lambda_bg': 1., 'lambda_missing': 2., 'lambda_cont': .2, 'missing_threshold': .45}
        sample = {'image': self.image[0], 'mask': self.mask[0], 'skeleton': self.skeleton[0]}
        with tempfile.TemporaryDirectory(prefix='.v9_smoke_test_', dir=ROOT) as tmp:
            args = SimpleNamespace(root_path=ROOT, device=torch.device('cpu'), seed=1234,
                                   batch_size=2, output_dir=Path(tmp))
            with patch.object(runner, 'RoadSkeletonDataset', return_value=[sample]):
                runner.smoke(args, self.model, source, {'source': 'synthetic'})
            report = json.loads((Path(tmp)/'smoke.json').read_text())
            self.assertTrue(report['passed'])
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0., rtol=0.)

    def test_preflight_refuses_missing_upstream(self):
        with tempfile.TemporaryDirectory(prefix='.v9_preflight_test_', dir=ROOT) as tmp:
            root = Path(tmp)
            v7 = root/'v7.pth'; v8 = root/'v8.pth'
            torch.save({'version': 'reliable_directional_dynamics_v7_continuity_transport',
                'modules': {name: {} for name in NAMES}, 'base_checkpoint': str(root/'missing_base.pth'),
                'q3_checkpoint': str(root/'missing_q3.pth')}, v7)
            torch.save({'version': 'reliable_directional_dynamics_v8_semantic_gate',
                        'v7_checkpoint': str(v7.resolve()), 'relation_mode': 'full', 'gate_relation': {}}, v8)
            args = SimpleNamespace(v7_checkpoint=v7, v8_checkpoint=v8, output_dir=root/'report')
            with self.assertRaises(SystemExit):
                runner.preflight(args)
            report = json.loads((root/'report/preflight.json').read_text())
            self.assertFalse(report['can_run_full_model'])
            self.assertFalse(report['checkpoints']['base']['exists'])

    def test_training_loop_and_resume(self):
        sample = {'image': self.image[0], 'mask': self.mask[0], 'skeleton': self.skeleton[0]}
        class Dataset:
            def __len__(self): return 2
            def __getitem__(self, unused): return sample
            def set_epoch(self, unused): pass
        source = {'lambda_bg': 1., 'lambda_missing': 2., 'lambda_cont': .2, 'missing_threshold': .45}
        identity = {'source': 'synthetic'}
        q3 = SimpleNamespace(full_dataset=lambda *unused: [sample])
        evaluation = SimpleNamespace(evaluate_thresholds=lambda *unused, **kwargs:
            ({.6: {'selection_score': .5}}, None, None, None))
        with tempfile.TemporaryDirectory(prefix='.v9_train_test_', dir=ROOT) as tmp:
            root = Path(tmp)
            (root/'stage0.json').write_text(json.dumps({'passed': True, 'identity': identity,
                'images': 1, 'root_path': str(ROOT.resolve()), 'reference_threshold': .45}))
            (root/'smoke.json').write_text(json.dumps({'passed': True, 'identity': identity}))
            args = SimpleNamespace(command='train', output_dir=root, root_path=ROOT,
                reference_threshold=.45, stage='B', previous_checkpoint=None,
                dynamic_checkpoint=None, seed=1234, batch_size=2, num_workers=0,
                device=torch.device('cpu'), epochs=1, val_interval=1, max_images=0)
            with patch.dict(sys.modules, {'run_q3_r3_mass': q3, 'run_reliable_directional_mass': evaluation}), \
                 patch.object(runner, 'RoadSkeletonDataset', return_value=Dataset()):
                runner.train(args, self.model, source, identity)
                self.assertTrue((root/'stage_B/best.pth').is_file())
                args.command = 'resume'; args.epochs = 2
                args.dynamic_checkpoint = root/'stage_B/last.pth'
                runner.train(args, self.model, source, identity)
            saved = torch.load(root/'stage_B/last.pth', weights_only=False)
            self.assertEqual(saved['epoch'], 2)
            self.assertEqual(saved['stage'], 'B')


if __name__ == '__main__':
    unittest.main()
