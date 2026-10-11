"""Read-only checkpoint probe for zero validation IoU."""

import argparse
import json
from pathlib import Path

import torch

from data1 import FullImageData1
from inference import counts, metrics, positions, predict_full
from networks.WeavingUnet import WeavingUnet


THRESHOLDS = (0.05, 0.1, 0.2, 0.5)


def describe(probability, target):
    probability = probability.detach().float().cpu()
    target = target.detach().bool().cpu()
    output = {
        "gt_pixels": int(target.sum()),
        "prob_mean": round(float(probability.mean()), 6),
        "prob_p99": round(float(torch.quantile(probability.flatten(), 0.99)), 6),
        "prob_max": round(float(probability.max()), 6),
    }
    for threshold in THRESHOLDS:
        tp, fp, fn = counts(probability, target, threshold)
        key = f"threshold_{threshold:g}"
        output[key] = {"pred_pixels": tp + fp, "tp": tp,
                       "iou": round(metrics(tp, fp, fn)["iou"], 6)}
    return output


def road_rich_tile(image, label, size=512):
    height, width = label.shape
    candidates = ((int(label[y:y + size, x:x + size].sum()), y, x)
                  for y in positions(height, size, size)
                  for x in positions(width, size, size))
    _, top, left = max(candidates)
    return image[:, top:top + size, left:left + size], label[top:top + size, left:left + size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {split: FullImageData1(args.data_root, split) for split in ("train", "val")}
    for checkpoint_name in ("best.pth", "last.pth"):
        path = args.run_dir / checkpoint_name
        if not path.is_file():
            print(json.dumps({"checkpoint": checkpoint_name, "error": "missing"}), flush=True)
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = WeavingUnet(pretrained_encoder=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(json.dumps({"checkpoint": checkpoint_name, "epoch": checkpoint.get("epoch"),
                          "best_val_iou": checkpoint.get("best_val_iou")}), flush=True)
        with torch.inference_mode():
            for split, dataset in datasets.items():
                samples = [dataset[i] for i in range(min(args.samples, len(dataset)))]
                model.eval()
                for case, image, label in samples:
                    probability = predict_full(model, image[None].to(device))[0, 0]
                    print(json.dumps({"checkpoint": checkpoint_name, "split": split,
                                      "mode": "eval_overlap", "case": case,
                                      **describe(probability, label)}), flush=True)
                if len(samples) >= 2:
                    tiles = [road_rich_tile(image, label) for _, image, label in samples]
                    image_batch = torch.stack([item[0] for item in tiles]).to(device)
                    label_batch = torch.stack([item[1] for item in tiles])
                    for mode in ("eval", "train"):
                        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                        model.eval() if mode == "eval" else model.train()
                        probabilities = model(image_batch)[:, 0]
                        for i, (case, _, _) in enumerate(samples):
                            print(json.dumps({"checkpoint": checkpoint_name, "split": split,
                                              "mode": f"{mode}_paired_tile", "case": case,
                                              **describe(probabilities[i], label_batch[i])}), flush=True)


if __name__ == "__main__":
    main()
