"""Measure the full and surface-only e202 model on the same GPU and inputs."""

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import get_config
from networks.vision_transformer import SwinUnet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain_ckpt", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if not args.pretrain_ckpt.is_file():
        parser.error(f"Missing pretrained checkpoint: {args.pretrain_ckpt}")
    if args.repeats < 1 or args.warmup < 0 or args.batch_size < 1:
        parser.error("Invalid repeat, warmup or batch count")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for synchronized GPU timing")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    options = SimpleNamespace(cfg=str(ROOT / "configs/swin_tiny_patch4_window7_224_lite.yaml"),
                              opts=None, batch_size=args.batch_size, img_size=args.img_size,
                              zip=False, cache_mode="", resume="", accumulation_steps=1,
                              use_checkpoint=False, amp_opt_level="", tag="", eval=False,
                              throughput=False)
    config = get_config(options)
    config.defrost()
    config.MODEL.PRETRAIN_CKPT = str(args.pretrain_ckpt)
    config.freeze()
    torch.manual_seed(1234)
    image = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device="cuda")
    measurements = {}
    for surface_only in (False, True):
        torch.manual_seed(1234)
        model = SwinUnet(config, num_classes=1, return_skeleton=True,
                         structure_profile="stage23_boundary_0626", stage_skeleton_mode="direct",
                         enable_highres_structure_stream=True, enable_e128_stage_fusion=True,
                         enable_h3_surface_fusion=True, remove_stage2_pre_topology_source=True,
                         enable_global_topology=True, surface_only=surface_only)
        model.load_from(config)
        parameters = sum(p.numel() for p in model.parameters())
        surface_parameters = sum(p.numel() for name, p in model.named_parameters()
                                 if any(name.startswith("swin_unet.guided_head." + part + ".")
                                        for part in ("surface_proj", "surface_branch", "surface_refine", "surface_head")))
        model.cuda().eval()
        samples = []
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(image)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                model(image)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
        key = "surface_only" if surface_only else "full_e202"
        measurements[key] = dict(parameters=parameters, shared_surface_head_parameters=surface_parameters,
                                 forward_batch_ms_mean=statistics.mean(samples),
                                 forward_batch_ms_median=statistics.median(samples),
                                 forward_ms_per_tile_mean=statistics.mean(samples) / args.batch_size,
                                 peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024 ** 2)
        print(json.dumps({key: measurements[key]}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    full, surface = measurements["full_e202"], measurements["surface_only"]
    measurements["comparison"] = dict(
        removed_parameters=full["parameters"] - surface["parameters"],
        surface_to_full_forward_time=surface["forward_batch_ms_mean"] / full["forward_batch_ms_mean"],
        shared_surface_head_parameter_counts_match=(full["shared_surface_head_parameters"] == surface["shared_surface_head_parameters"]))
    report = dict(gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, precision="FP32",
                  img_size=args.img_size, batch_size=args.batch_size, warmup=args.warmup,
                  repeats=args.repeats, pretrained_checkpoint=str(args.pretrain_ckpt),
                  timing_scope="synchronized complete model forward on resident GPU input, including all enabled branches; excludes H2D, disk, sliding fusion, metrics and training",
                  measurements=measurements)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"BENCHMARK={args.output_json}", flush=True)


if __name__ == "__main__":
    main()
