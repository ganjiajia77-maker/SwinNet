"""Compare v6 all-node APLS with v7 indexed insertion and sampled controls."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import time

from compare_predictions import (label_index, prediction_index, read_binary,
                                 resolve_path, threshold_of)
from official_graph_metrics import (_official_apls, prepare_apls_reference,
                                    prepare_official_graph, _cosmiq)
from road_metrics import DEFAULT_PROTOCOL, skeleton_graph, skeletonize


def load_old(path):
    spec = importlib.util.spec_from_file_location("previous_graph_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--old_evaluator", type=Path,
                        help="v6 evaluation/road_comparison/official_graph_metrics.py")
    parser.add_argument("--max_models", type=int, default=4)
    parser.add_argument("--sample_controls", type=int, default=500)
    args = parser.parse_args()
    if args.max_models < 1 or args.sample_controls < 1:
        parser.error("--max_models and --sample_controls must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    base = args.manifest.parent
    protocol = dict(DEFAULT_PROTOCOL)
    protocol.update({k: v for k, v in manifest.get("protocol", {}).items()
                     if k in protocol})
    root = resolve_path(manifest["data_root"], base)
    labels = label_index(root, "test")
    if args.case not in labels:
        parser.error(f"Unknown test case {args.case}")

    target = read_binary(labels[args.case], protocol["image_size"],
                         allow_replicated_rgb=True)
    points, csr = skeleton_graph(skeletonize(target))
    ground = prepare_official_graph(points, csr)
    proposals = []
    for model in manifest["models"][:args.max_models]:
        name = model["name"]
        threshold = selection["models"][name]["threshold"]
        sources = [item for item in model["predictions"]["test"]
                   if abs(threshold_of(item) - threshold) < 1e-12]
        if len(sources) != 1:
            parser.error(f"Expected one selected threshold for {name}")
        predictions = prediction_index(sources[0], base, labels)
        mask = read_binary(predictions[args.case], protocol["image_size"])
        pred_points, pred_csr = skeleton_graph(skeletonize(mask))
        proposals.append((name, prepare_official_graph(
            pred_points, pred_csr, id_offset=1_000_000_000)))

    old_scores = {}
    old_total = None
    if args.old_evaluator:
        previous = load_old(args.old_evaluator)
        previous._cosmiq()
        _cosmiq()
        started = time.perf_counter()
        previous.prepare_apls_reference(ground)
        old_total = time.perf_counter() - started
        print(f"[APLS-BENCH] v6 shared_GT={old_total:.1f}s", flush=True)
        for name, proposal in proposals:
            started = time.perf_counter()
            total, _, _, _, _ = previous._official_apls(ground, proposal, protocol)
            elapsed = time.perf_counter() - started
            old_total += elapsed
            old_scores[name] = total
            print(f"[APLS-BENCH] v6 {args.case} {name} score={total:.6f} "
                  f"seconds={elapsed:.1f}", flush=True)
        print(f"[APLS-BENCH] v6 total={old_total:.1f}s", flush=True)

    def run_variant(label, limit):
        variant = dict(protocol, apls_max_control_nodes=limit)
        started = time.perf_counter()
        reference = prepare_apls_reference(ground, variant)
        gt_elapsed = time.perf_counter() - started
        total_seconds = gt_elapsed
        print(f"[APLS-BENCH] {label} shared_GT={gt_elapsed:.1f}s "
              f"controls={len(reference['controls'])}", flush=True)
        for name, proposal in proposals:
            started = time.perf_counter()
            score, _, _, counts, stages = _official_apls(ground, proposal, variant)
            elapsed = time.perf_counter() - started
            total_seconds += elapsed
            if limit == 0 and name in old_scores and not math.isclose(
                    score, old_scores[name], rel_tol=0, abs_tol=1e-10):
                raise AssertionError(f"Full-control score changed for {name}: "
                                     f"v6={old_scores[name]}, fast={score}")
            print(f"[APLS-BENCH] {label} {args.case} {name} score={score:.6f} "
                  f"seconds={elapsed:.1f} controls_pred={counts['pred_controls']} "
                  f"pred_native={stages['pred_native']:.1f} "
                  f"insert={stages['control_insert']:.1f} "
                  f"prime_paths={stages['prime_paths']:.1f} "
                  f"score_phase={stages['score']:.1f}", flush=True)
        print(f"[APLS-BENCH] {label} total={total_seconds:.1f}s", flush=True)
        if old_total is not None:
            print(f"[APLS-BENCH] {label} speedup_vs_v6={old_total / total_seconds:.2f}x",
                  flush=True)
        return total_seconds

    run_variant("v7_fast_full", 0)
    run_variant(f"v7_sampled_{args.sample_controls}", args.sample_controls)


if __name__ == "__main__":
    main()
