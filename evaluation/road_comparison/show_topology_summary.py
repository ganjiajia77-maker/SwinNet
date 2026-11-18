"""Display the six graph/mask topology metrics from an existing comparison CSV."""

import argparse
import csv
from pathlib import Path


FIELDS = (
    ("clDice", "cldice"),
    ("TOPO-P", "topo_precision"),
    ("TOPO-R", "topo_recall"),
    ("TOPO-F1", "topo_f1"),
    ("APLS", "apls_bidirectional"),
    ("TLTS-Infeasible", "tlts_infeasible"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="test_comparison.csv")
    args = parser.parse_args()
    with args.csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"name", *(key for _, key in FIELDS)}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            parser.error("CSV lacks required fields: " + ", ".join(sorted(missing)))
        rows = list(reader)
    if not rows:
        parser.error("CSV has no model results")
    for row in rows:
        values = "  ".join(f"{label}={float(row[key]):.6f}" for label, key in FIELDS)
        print(f"{row['name']}: {values}")


if __name__ == "__main__":
    main()
