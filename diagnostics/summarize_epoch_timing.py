"""Summarize the measured training-batch time in existing training CSV logs."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def summarize(args):
    batches = {}
    duplicates = 0
    invalid_rows = 0
    for source in args.batch_csv:
        with Path(source).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"epoch", "batch", "ms_per_batch"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{source}: required columns are {sorted(required)}")
            for row in reader:
                try:
                    epoch, batch = int(row["epoch"]), int(row["batch"])
                except (TypeError, ValueError):
                    invalid_rows += 1
                    continue
                milliseconds = finite_number(row.get("ms_per_batch"))
                if milliseconds is None or milliseconds < 0 or epoch < 1 or batch < 1:
                    invalid_rows += 1
                    continue
                if epoch < args.start_epoch or (args.end_epoch and epoch > args.end_epoch):
                    continue
                key = (epoch, batch)
                if key in batches:
                    duplicates += 1
                batches[key] = {
                    "batch": batch,
                    "milliseconds": milliseconds,
                    "stage2_active_ratio": finite_number(row.get("stage2_active_ratio")),
                    "stage3_active_ratio": finite_number(row.get("stage3_active_ratio")),
                }
    if not batches:
        raise ValueError("No valid timing rows in the requested epoch range.")

    epochs = defaultdict(list)
    for (epoch, _), row in batches.items():
        epochs[epoch].append(row)
    records = []
    for epoch, rows in sorted(epochs.items()):
        batch_ids = {row["batch"] for row in rows}
        complete = (
            batch_ids == set(range(1, args.expected_batches + 1))
            if args.expected_batches else None
        )
        milliseconds = [row["milliseconds"] for row in rows]
        record = {
            "epoch": epoch,
            "phase": "routing_warmup" if epoch <= args.warmup_epochs else "after_warmup",
            "logged_batches": len(rows),
            "complete": complete,
            "train_batch_seconds": sum(milliseconds) / 1000.0,
            "train_batch_minutes": sum(milliseconds) / 60000.0,
            "mean_ms_per_batch": mean(milliseconds),
        }
        for stage in (2, 3):
            key = f"stage{stage}_active_ratio"
            values = [row[key] for row in rows if row[key] is not None]
            record[key] = mean(values) if values else None
        records.append(record)

    eligible = [row for row in records if row["complete"] is not False]
    groups = {}
    for name in ("all", "routing_warmup", "after_warmup"):
        selected = [row for row in eligible if name == "all" or row["phase"] == name]
        if not selected:
            groups[name] = None
            continue
        groups[name] = {
            "first_epoch": selected[0]["epoch"],
            "last_epoch": selected[-1]["epoch"],
            "n_epochs": len(selected),
            "mean_train_batch_seconds_per_epoch": mean(row["train_batch_seconds"] for row in selected),
            "mean_train_batch_minutes_per_epoch": mean(row["train_batch_minutes"] for row in selected),
            "mean_ms_per_batch": sum(row["train_batch_seconds"] for row in selected) * 1000.0
            / sum(row["logged_batches"] for row in selected),
        }
        for stage in (2, 3):
            key = f"stage{stage}_active_ratio"
            values = [row[key] for row in selected if row[key] is not None]
            groups[name][key] = mean(values) if values else None

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "timing_scope": "Sum of synchronized ms_per_batch, not full epoch wall time.",
        "excluded_costs": "DataLoader wait, post-timing logging, validation, routing calibration, checkpoint I/O.",
        "duplicate_rows_replaced_by_later_rows": duplicates,
        "invalid_rows_skipped": invalid_rows,
        "expected_batches": args.expected_batches or None,
        "excluded_incomplete_epochs": [row["epoch"] for row in records if row["complete"] is False],
        "groups": groups,
        "output_csv": str(output.resolve()),
    }
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_csv", nargs="+", required=True, help="Logs in chronological order; later rows replace duplicate epoch/batch keys.")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--end_epoch", type=int, default=0)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--expected_batches", type=int, default=0, help="Exclude incomplete epochs from averages; 0 disables completeness checking.")
    args = parser.parse_args()
    if args.start_epoch < 1 or args.expected_batches < 0 or args.warmup_epochs < 0:
        parser.error("Invalid epoch or batch-count settings.")
    if args.end_epoch and args.end_epoch < args.start_epoch:
        parser.error("end_epoch must be at least start_epoch.")
    summarize(args)


if __name__ == "__main__":
    main()
