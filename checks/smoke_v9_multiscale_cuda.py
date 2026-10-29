"""512-input CUDA smoke with a synthetic frozen provider, actual V9 downstream."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from checks.test_v9_state_gate import model as original_model, repair
from networks.v9_multiscale_transport import install
import run_v9_multiscale as runner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'validation/multiscale_cuda.json')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; this GPU check was not executed')
    torch.set_num_threads(2)
    results = []
    for batch in (1, 2):
        torch.manual_seed(1234)
        model = original_model().cuda().eval()
        image = torch.randn(batch, 3, 512, 512, device='cuda')
        with torch.inference_mode():
            before = model(image)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        install(model, 'local')
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
        with torch.inference_mode():
            after = model(image)
        error = float((before['z1']-after['z1']).abs().max())
        assert error == 0.
        install(model, 'multiscale')
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
        model.train(); model.dynamics.blend = 1.
        optimizer = runner.configure(model, {name: 1e-4 for name in runner.v9.STAGE_MODULES['D']}, 1e-4)
        mask = torch.zeros(batch, 1, 512, 512, device='cuda'); mask[:, :, 250:255, 20:490] = 1.
        skel = torch.zeros_like(mask); skel[:, :, 252, 20:490] = 1.
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.perf_counter()
        values = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            result = model(image, capture_diagnostics=True)
            loss, _ = repair.dynamic_losses(result, mask, skel)
            assert torch.isfinite(loss)
            loss.backward(); runner.check_gradients(model)
            grad = runner.v9.grad_norm(model.dynamics.selector)
            assert grad > 0.
            assert result['p1'].shape == (batch, 1, 512, 512)
            assert result['selector_weights'].shape == (batch, 3, 33, 128, 128)
            values.append({'loss': float(loss.detach()), 'selector_grad_l2': grad})
            optimizer.step()
            print(f'[CUDA] batch={batch} step={step+1}/2 passed', flush=True)
        torch.cuda.synchronize()
        results.append({'batch_size': batch, 'local_exact_logit_error': error, 'steps': values,
                        'seconds': time.perf_counter()-start,
                        'peak_mib': torch.cuda.max_memory_allocated()/2**20})
        del model, optimizer, before, after, result, loss
        torch.cuda.empty_cache()
    runner.write_json(args.output, {'passed': True, 'torch': torch.__version__,
        'device': torch.cuda.get_device_name(), 'results': results,
        'limit': 'Synthetic frozen BASE/Q3 provider. Actual upstream weights/data require server smoke.'})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
