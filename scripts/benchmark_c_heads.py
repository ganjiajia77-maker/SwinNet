"""Measure only the two alternative C heads; connectivity_context is shared."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from networks.skeleton_guided_head import PairwiseConnectivityHead


def measure(head, feature, direction, skeleton, warmup, repeats):
    head.eval()
    samples = []
    with torch.inference_mode():
        for iteration in range(warmup + repeats):
            if feature.is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            if isinstance(head, PairwiseConnectivityHead):
                output = head(feature, head.direction_alignment(direction), skeleton)
            else:
                output = head(feature)
            if feature.is_cuda:
                torch.cuda.synchronize()
            if iteration >= warmup:
                samples.append((time.perf_counter() - start) * 1000)
    return dict(params=sum(p.numel() for p in head.parameters()),
                output_shape=list(output.shape), median_ms=statistics.median(samples),
                mean_ms=statistics.mean(samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--spatial_size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    feature = torch.randn(args.batch_size, 96, args.spatial_size, args.spatial_size, device=args.device)
    direction = torch.randn(args.batch_size, 2, args.spatial_size, args.spatial_size, device=args.device)
    skeleton = torch.rand(args.batch_size, 1, args.spatial_size, args.spatial_size, device=args.device)
    pair = PairwiseConnectivityHead(96).to(args.device)
    conv = torch.nn.Conv2d(96, 8, 1).to(args.device)
    result = dict(device=args.device, dtype="fp32", batch_size=args.batch_size,
                  spatial_size=args.spatial_size, warmup=args.warmup, repeats=args.repeats,
                  scope="C head only; excludes shared connectivity_context and full model",
                  pairwise=measure(pair, feature, direction, skeleton, args.warmup, args.repeats),
                  conv=measure(conv, feature, direction, skeleton, args.warmup, args.repeats))
    result["pairwise_over_conv_median"] = (result["pairwise"]["median_ms"] /
                                           result["conv"]["median_ms"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
