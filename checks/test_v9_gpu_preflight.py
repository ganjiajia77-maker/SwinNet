"""No GPU or torch needed: physical selection and occupied-card handling."""
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gpu_preflight', ROOT/'scripts/v9_gpu_preflight.py')
gpu = importlib.util.module_from_spec(spec); spec.loader.exec_module(gpu)


class PreflightTests(unittest.TestCase):
    def test_uuid_mapping_filters_other_gpus(self):
        outputs = ['2, GPU-target, Quadro RTX 5000, 16128, 128, 16000\n',
                   '42883, GPU-other, 13486, python\n123, GPU-target, 64, python\n']
        report = io.StringIO()
        with patch.object(gpu.subprocess, 'check_output', side_effect=outputs) as call:
            self.assertEqual(gpu.preflight('2', 6144, report), 'GPU-target')
        self.assertEqual(call.call_args_list[0].args[0][1:3], ['-i', '2'])
        self.assertIn('logical cuda:0', report.getvalue())
        self.assertIn('pid=123', report.getvalue())
        self.assertNotIn('pid=42883', report.getvalue())

    def test_occupied_card_stops_without_killing(self):
        outputs = ['2, GPU-target, Quadro RTX 5000, 16128, 14000, 2128\n',
                   '42883, GPU-target, 13486, python\n']
        report = io.StringIO()
        with patch.object(gpu.subprocess, 'check_output', side_effect=outputs) as call:
            with self.assertRaisesRegex(RuntimeError, 'only 2128MiB'):
                gpu.preflight('2', 6144, report)
        self.assertIn('pid=42883', report.getvalue())
        self.assertTrue(all(c.args[0][0] == 'nvidia-smi' for c in call.call_args_list))

    def test_missing_or_ambiguous_gpu_is_rejected(self):
        for output in ('', '2, no-uuid, Quadro, 16128, 128, 16000\n',
                       '0, GPU-a, A, 16, 0, 16\n1, GPU-b, B, 16, 0, 16\n'):
            with patch.object(gpu.subprocess, 'check_output', return_value=output):
                with self.assertRaises(ValueError): gpu.query_gpu('2')


if __name__ == '__main__':
    unittest.main()
