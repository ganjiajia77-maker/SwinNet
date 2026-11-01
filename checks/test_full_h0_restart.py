"""Check recovery against saved run directories without starting a GPU job."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT=Path(__file__).resolve().parents[1]
SCRIPT=(ROOT/'scripts/restart_full_h0_B_identity_fix.sh').read_text()
CODE=SCRIPT.split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]


class RestartTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT,prefix='.restart_test_')
        self.addCleanup(self.temp.cleanup)
        self.pair=Path(self.temp.name)
        (self.pair/'A/eval').mkdir(parents=True)
        (self.pair/'pair_contract.json').write_text(json.dumps({'epochs':20}))
        (self.pair/'A/training_complete.json').write_text(json.dumps({'epochs':20}))
        (self.pair/'A/eval/val_selection.json').write_text('{}')
        (self.pair/'A/best.pth').write_bytes(b'preserve A weights')

    def run_script(self):
        return subprocess.run([sys.executable,'-c',CODE,str(self.pair)],capture_output=True,text=True)

    def test_archives_only_failed_B_and_preserves_A(self):
        (self.pair/'B').mkdir()
        (self.pair/'B/experiment.json').write_text(json.dumps({'rate_initial_bias':-8.}))
        result=self.run_script()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.strip(),'20')
        self.assertFalse((self.pair/'B').exists())
        self.assertEqual(len(list(self.pair.glob('B_before_identity_fix_*'))),1)
        self.assertEqual((self.pair/'A/best.pth').read_bytes(),b'preserve A weights')

    def test_refuses_to_replace_existing_old_B_checkpoint(self):
        (self.pair/'B').mkdir()
        (self.pair/'B/last.pth').write_bytes(b'preserve B weights')
        result=self.run_script()
        self.assertNotEqual(result.returncode,0)
        self.assertEqual((self.pair/'B/last.pth').read_bytes(),b'preserve B weights')

    def test_fixed_B_can_resume_without_archiving(self):
        (self.pair/'B').mkdir()
        (self.pair/'B/experiment.json').write_text(json.dumps({'rate_initial_bias':-13.,
            'rate_initialization':'final rates output weight=0; bias=[2.5,-13,-13,1]'}))
        (self.pair/'B/last.pth').write_bytes(b'fixed B weights')
        result=self.run_script()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual((self.pair/'B/last.pth').read_bytes(),b'fixed B weights')
        self.assertFalse(list(self.pair.glob('B_before_identity_fix_*')))

    def test_incomplete_A_is_not_restarted(self):
        (self.pair/'A/eval/val_selection.json').unlink()
        result=self.run_script()
        self.assertNotEqual(result.returncode,0)
        self.assertFalse((self.pair/'B').exists())


if __name__=='__main__':
    unittest.main()
