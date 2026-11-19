"""Run directly: python tests/check_scgra_export.py (no pytest dependency)."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'evaluation/road_comparison'))
from diagnostics.scgra_export import (grid_to_pixels, point_metrics, prepare_gt,
    selection_controls, trace_module, zero_global_alpha, paired_image)
from networks.keypoint_global_topology import KeypointGuidedGlobalTopology
from road_metrics import skeletonize
from scripts.export_scgra_decoder_mass import METRIC_FIELDS, aggregate, dump, main, select_validation_record
from diagnostics.scgra_decoder_mass import file_digest
from diagnostics.scgra_decoder_mass import NAME, validate_saved, discover_checkpoint
from h0_location import REFERENCE, DEFAULTS, BASE_COMMIT


class Checks(unittest.TestCase):
    def test_checkpoint_discovery_deduplicates_and_rejects_ambiguity(self):
        with TemporaryDirectory(dir=ROOT/'tests') as temporary:
            root = Path(temporary)
            first = root/(NAME+'_1234')/'best.pth'
            first.parent.mkdir()
            first.write_bytes(b'marker')
            dump(root/'h0_stage2_location_v1_decoder_latest.json', dict(
                architecture_id='h0_stage2_location_v1', stage2_structure_location='decoder', best=str(first)))
            self.assertEqual(discover_checkpoint(root), first.resolve())
            other = root/(NAME+'_5678')/'best.pth'
            other.parent.mkdir()
            other.write_bytes(b'marker')
            with self.assertRaisesRegex(ValueError, 'Candidates'):
                discover_checkpoint(root)

    def test_original_decoder_control_identity_rejects_other_experiments(self):
        saved = dict(REFERENCE, **DEFAULTS, architecture_id='h0_stage2_location_v1',
                     stage2_structure_location='decoder')
        contract = dict(architecture_id='h0_stage2_location_v1', stage2_structure_location='decoder',
                        base_model_commit=BASE_COMMIT)
        validate_saved(saved, contract)
        for changed in ({'global_topology_stage2': True}, {'encoder_h_postmerge_phase': True},
                        {'use_ema': True}, {'source_patch_size': 1024}, {'stage2_structure_location': 'encoder_e1'}):
            with self.assertRaises(ValueError):
                validate_saved(dict(saved, **changed), contract)
        with self.assertRaises(ValueError):
            validate_saved(saved, contract, has_ema=True)

    def test_unified_selection_requires_decoder_name_checkpoint_and_identity(self):
        from scripts.export_scgra_decoder_mass import digest
        with TemporaryDirectory(dir=ROOT/'tests') as temporary:
            root = Path(temporary)
            (root/'unified/val').mkdir(parents=True)
            selection = root/'unified/val/val_selection.json'
            provenance = dict(checkpoint_sha256='correct', weights='raw; no EMA')
            model = dict(name=NAME, provenance=provenance,
                         predictions={'val': [dict(threshold=.45, path='/deleted/validation/masks')]})
            manifest = root/'unified/comparison.json'
            dump(manifest, {'models': [model]})
            record = dict(selected_on='val', criterion='global_iou', models={NAME: dict(threshold=.45,
                          identity=digest(dict(name=NAME, provenance=provenance)))})
            dump(selection, record)
            self.assertEqual(select_validation_record(root/'best.pth', 'correct', None)[2], .45)
            with self.assertRaises(ValueError):
                select_validation_record(root/'best.pth', 'different checkpoint', None)
            record['models'][NAME]['identity'] = 'wrong identity'
            dump(selection, record)
            with self.assertRaises(ValueError):
                select_validation_record(root/'best.pth', 'correct', None)

    def test_capture_bit_identical_real_attention_and_effective_residual(self):
        torch.manual_seed(12)
        module = KeypointGuidedGlobalTopology(8, 4, max_nodes=4, heads=2, enabled=True).eval()
        module.raw_alpha.data.fill_(-.3)
        x, z, p, c = torch.randn(1, 8, 8, 10), torch.randn(1, 4, 8, 10), torch.rand(1, 1, 8, 10), torch.rand(1, 8, 8, 10)
        before = {k: v.clone() for k, v in module.state_dict().items()}
        with torch.inference_mode():
            normal = module.forward_feature_anchors(x, z, p, c)
            with trace_module(module):
                observed = module.forward_feature_anchors(x, z, p, c)
                self.assertTrue(torch.equal(normal, observed))
                trace = module.last_scgra_trace
                self.assertEqual(module._scgra_trace_calls, 1)
                self.assertEqual(tuple(trace['node_attention'].shape), (1, 2, 4, 4))
                torch.testing.assert_close(trace['node_attention'].sum(-1), torch.ones(1, 2, 4))
                expected = (observed-x).square().mean(1).sqrt()
                torch.testing.assert_close(trace['effective_residual_rms'], expected, rtol=0, atol=0)
                torch.testing.assert_close(trace['gated_delta_rms'], trace['ungated_delta_rms']*p[:, 0], rtol=1e-5, atol=1e-7)
                selection = module.last_anchor_selection
                a, b, random, _ = selection_controls(module, selection, trace, 1234, 'case', 0, 0, 3)
                self.assertEqual([len(a), len(b), *[len(r) for r in random]], [4]*5)
                for points in random:
                    self.assertEqual(len(np.unique(points, axis=0)), 4)
                again = selection_controls(module, selection, trace, 1234, 'case', 0, 0, 3)[2]
                for first, second in zip(random, again):
                    np.testing.assert_array_equal(first, second)
            self.assertFalse(module.capture_scgra_trace)
            self.assertFalse(hasattr(module, '_scgra_trace_calls'))
            with zero_global_alpha(module):
                off = module.forward_feature_anchors(x, z, p, c)
                self.assertTrue(torch.equal(off, x))
            self.assertTrue(torch.equal(module.forward_feature_anchors(x, z, p, c), normal))
        self.assertEqual(before.keys(), module.state_dict().keys())
        for key, value in module.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_empty_anchors_zero_gate_and_matched_empty_controls(self):
        module = KeypointGuidedGlobalTopology(8, 4, max_nodes=4, heads=2, enabled=True).eval()
        module.raw_alpha.data.fill_(.4)
        x = torch.randn(1, 8, 4, 4)
        with torch.inference_mode(), trace_module(module):
            output = module.forward_feature_anchors(x, torch.zeros(1, 4, 4, 4), torch.zeros(1, 1, 4, 4))
            self.assertTrue(torch.equal(output, x))
            self.assertFalse(module.last_anchor_selection['selected_valid'].any())
            control = selection_controls(module, module.last_anchor_selection, module.last_scgra_trace, 1, 'empty', 0, 0, 2)
            self.assertEqual([len(control[0]), len(control[1]), *[len(r) for r in control[2]]], [0]*4)

    def test_original_pixel_tolerance_component_coverage_and_deduplication(self):
        gt = np.zeros((20, 20), bool)
        gt[4, 2:8] = True
        gt[15, 12:18] = True
        labels, lengths, distance, _ = prepare_gt(gt, skeletonize)
        points = np.array([[4, 3], [4, 3], [3, 3], [15, 14]])
        result = point_metrics(points, gt, labels, lengths, distance, 1.)
        self.assertEqual(result['anchor_events'], 4)
        self.assertEqual(result['unique_anchor_locations'], 3)
        self.assertAlmostEqual(result['road_hit_rate_exact'], 2/3)
        self.assertEqual(result['road_hit_rate_tolerant'], 1.)
        self.assertEqual(result['component_coverage'], 1.)
        self.assertEqual(result['skeleton_length_weighted_component_coverage'], 1.)
        coords = grid_to_pixels([[0, 0], [127, 127]], (128, 128), (512, 512), (256, 512))
        np.testing.assert_array_equal(coords, [[258, 514], [766, 1022]])
        empty = point_metrics([], gt, labels, lengths, distance, 1.)
        self.assertIsNone(empty['road_hit_rate_exact'])
        self.assertEqual(empty['component_coverage'], 0.)

    def test_validation_threshold_identity_and_ambiguity(self):
        with TemporaryDirectory(dir=ROOT/'tests') as temporary:
            root = Path(temporary)
            checkpoint = root/'best.pth'
            first, second = root/'one/val_selection.json', root/'two/val_selection.json'
            first.parent.mkdir()
            second.parent.mkdir()
            record = dict(selected_on='val', criterion='global_iou', checkpoint_sha256='correct', threshold=.48, eval_weights='raw')
            dump(first, record)
            dump(second, dict(record, checkpoint_sha256='wrong', threshold=.5))
            self.assertEqual(select_validation_record(checkpoint, 'correct', None)[2], .48)
            dump(second, dict(record, threshold=.5))
            with self.assertRaisesRegex(ValueError, 'Multiple'):
                select_validation_record(checkpoint, 'correct', None)
            self.assertEqual(select_validation_record(checkpoint, 'correct', first)[2], .48)
            dump(first, dict(record, selected_on='test'))
            with self.assertRaisesRegex(ValueError, 'No matching'):
                select_validation_record(checkpoint, 'correct', first)

    def test_random_repeats_are_averaged_within_image_first(self):
        def row(case, repeat, value):
            return dict(case_id=case, method='random', repeat=repeat,
                **{field: value for field in METRIC_FIELDS})
        records = [dict(image_rows=[row('a', 0, 0.), row('a', 1, 1.)]), dict(image_rows=[row('b', 0, 1.)])]
        _, summary = aggregate(records)
        random = next(r for r in summary if r['method'] == 'random')
        self.assertEqual(random['images'], 2)
        self.assertEqual(random['mean_road_hit_rate_exact'], .75)

    def test_cli_metadata_writing_and_resume_without_repeating_forward(self):
        with TemporaryDirectory(dir=ROOT/'tests') as temporary:
            root = Path(temporary)
            run, images, labels = root/'run', root/'data/val/image', root/'data/val/mask'
            for p in (run/'val', images, labels):
                p.mkdir(parents=True)
            checkpoint = run/'best.pth'
            checkpoint.write_bytes(b'checkpoint marker for mocked loader')
            (images/'case_sat.jpg').write_bytes(b'RGB marker for input hash')
            (labels/'case_mask.png').write_bytes(b'GT marker for input hash')
            dump(run/'val/val_selection.json', dict(selected_on='val', criterion='global_iou',
                checkpoint_sha256=file_digest(checkpoint), threshold=.48, eval_weights='raw'))
            class Data:
                image_files = ['case_sat.jpg']
                label_files = ['case_mask.png']
                image_dir, mask_dir = str(images), str(labels)
                def __len__(self):return 1
            model = torch.nn.Module()
            model.swin_unet = torch.nn.Module()
            model.swin_unet.global_topology = KeypointGuidedGlobalTopology(8, 4, 4, 2, enabled=True)
            out = root/'export'
            calls = []
            def process(args, *unused):
                calls.append(1)
                artifact = args.output_dir/'cases/case/window_info.json'
                artifact.parent.mkdir(parents=True, exist_ok=True)
                dump(artifact, {'read_only': True})
                rows = [dict(case_id='case', method=method, repeat=0,
                    **{field: .5 for field in METRIC_FIELDS}) for method in ('structure_fps', 'probability_fps', 'random')]
                return dict(case_id='case', capture_verified=True, image_rows=rows,
                    window_rows=[dict(case_id='case', window_index=0)], effects=dict(case_id='case', paired=True),
                    artifacts={str(artifact.relative_to(out)): file_digest(artifact)})
            argv = ['export_scgra_decoder_mass.py', '--checkpoint', str(checkpoint), '--root_path', str(root/'data'),
                '--output_dir', str(out), '--max_images', '1', '--export_count', '1', '--device', 'cpu', '--paired_all']
            with patch('scripts.export_scgra_decoder_mass.evaluate.dataset', return_value=Data()), \
                    patch('scripts.export_scgra_decoder_mass.evaluate.load_model', return_value=(model, {'stage2_structure_location': 'decoder'}, 80)), \
                    patch('scripts.export_scgra_decoder_mass.process_image', side_effect=process), patch.object(sys, 'argv', argv):
                main()
                self.assertTrue((out/'export_complete.json').is_file())
                self.assertTrue((out/'anchor_summary.csv').is_file())
                self.assertEqual(len(calls), 1)
                main()
                self.assertEqual(len(calls), 1)
                (out/'cases/case/window_info.json').unlink()
                main()
                self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
