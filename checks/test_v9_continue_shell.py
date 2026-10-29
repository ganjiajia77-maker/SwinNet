"""Exercise the real Bash controller with a fake Python; never trains a model."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash') or ('D:/Git/bin/bash.exe' if Path('D:/Git/bin/bash.exe').is_file() else None)
FAKE = '''#!/usr/bin/env bash
set -e
if [[ "$1" == scripts/v9_gpu_preflight.py ]]; then
  [[ ${TEST_GPU_BUSY:-0} != 1 ]] || { echo 'test occupied GPU' >&2; exit 2; }
  echo GPU-test
else
  printf '%s\\n' "$*" >> "$TEST_DRIVER_LOG"
fi
'''


@unittest.skipUnless(BASH, 'Bash unavailable')
class ContinueShellTests(unittest.TestCase):
    def setup_fixture(self, root):
        run = root/'source'; run.mkdir(); source = run/'source.pth'; source.write_bytes(b'source')
        output = root/'pair'; output.mkdir()
        python = root/'fake-python'; python.write_text(FAKE, encoding='utf-8'); python.chmod(0o755)
        log = root/'calls.txt'
        env = dict(os.environ, RUN_DIR=run.as_posix(), SOURCE_CHECKPOINT=source.as_posix(),
                   MS_ROOT=output.as_posix(), PYTHON=python.as_posix(), TEST_DRIVER_LOG=log.as_posix(), GPU='2')
        return output, log, env

    def test_resume_checkpoint_and_preserve_incomplete_then_finish_pipeline(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            out, log, env = self.setup_fixture(Path(directory))
            (out/'local').mkdir(); (out/'local/last.pth').write_bytes(b'keep-this')
            (out/'multiscale').mkdir(); (out/'multiscale/experiment.json').write_text('{}')
            result = subprocess.run([BASH, 'scripts/run_v9_multiscale.sh', 'continue_all'],
                                    cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            calls = log.read_text().splitlines()
            self.assertEqual(len(calls), 9)
            self.assertIn('-u run_v9_multiscale.py resume ', calls[1])
            self.assertIn('--transport_mode local', calls[1])
            self.assertIn('-u run_v9_multiscale.py train ', calls[2])
            self.assertIn('--transport_mode multiscale', calls[2])
            self.assertIn('--audit_split test', calls[-1])
            self.assertEqual((out/'local/last.pth').read_bytes(), b'keep-this')
            backups = list(out.glob('multiscale_interrupted_no_checkpoint_*'))
            self.assertEqual(len(backups), 1)
            self.assertTrue((backups[0]/'experiment.json').is_file())

    def test_busy_card_stops_before_resume_or_directory_move(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            out, log, env = self.setup_fixture(Path(directory))
            (out/'multiscale').mkdir(); (out/'multiscale/experiment.json').write_text('{}')
            env['TEST_GPU_BUSY'] = '1'
            result = subprocess.run([BASH, 'scripts/run_v9_multiscale.sh', 'continue_all'],
                                    cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(log.exists())
            self.assertTrue((out/'multiscale/experiment.json').is_file())
            self.assertFalse(list(out.glob('*interrupted*')))


if __name__ == '__main__':
    unittest.main()
