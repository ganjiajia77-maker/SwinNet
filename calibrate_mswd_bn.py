"""Recover MSWD inference statistics using fixed, label-free training crops only."""

import argparse
import json
from pathlib import Path
import random
import time

import cv2
import torch
from torch.utils.data import DataLoader, Dataset

from bn_recovery import METHOD, load_checkpoint_model, split_mswd_stats
from data1 import normalize
from networks.WeavingUnet import WeavingUnet


class CalibrationCrops(Dataset):
    def __init__(self, root, samples=256, seed=1234):
        directory = Path(root) / "train" / "image"
        images = sorted(p for p in directory.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
        if not images:
            raise ValueError(f"No training images under {directory}")
        rng = random.Random(seed)
        images = rng.sample(images, min(samples, len(images)))
        self.records = []
        for path in images:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or min(image.shape[:2]) < 512:
                raise ValueError(f"Unreadable or smaller than 512: {path}")
            height, width = image.shape[:2]
            self.records.append({"image": str(path.resolve()),
                                 "top": rng.randint(0, height - 512),
                                 "left": rng.randint(0, width - 512)})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {record['image']}")
        y, x = record["top"], record["left"]
        return normalize(image[y:y + 512, x:x + 512])


class ChannelMoments:
    def __init__(self):
        self.count = self.batches = 0
        self.mean = self.m2 = None

    def update(self, x):
        if x.ndim != 4 or not torch.isfinite(x).all():
            raise FloatingPointError("Non-finite or invalid BN input during calibration")
        variance, mean = torch.var_mean(x.float(), dim=(0, 2, 3), correction=0)
        mean, variance = mean.double(), variance.double()
        count = x.shape[0] * x.shape[2] * x.shape[3]
        if self.count == 0:
            self.mean, self.m2 = mean, variance * count
        else:
            delta = mean - self.mean
            total = self.count + count
            self.mean = self.mean + delta * (count / total)
            self.m2 = self.m2 + variance * count + delta.square() * (self.count * count / total)
        self.count += count
        self.batches += 1

    def apply(self, bn):
        if self.count < 2:
            raise ValueError("Not enough observations to calibrate BN")
        variance = (self.m2 / (self.count - 1)).clamp_min(0)
        if not torch.isfinite(self.mean).all() or not torch.isfinite(variance).all():
            raise FloatingPointError("Non-finite calibrated BN statistics")
        with torch.no_grad():
            bn.running_mean.copy_(self.mean.to(bn.running_mean))
            bn.running_var.copy_(variance.to(bn.running_var))
            bn.num_batches_tracked.fill_(self.batches)


class StopAtNorm(Exception):
    pass


@torch.inference_mode()
def calibrate_layerwise(model, loader, targets, device, progress_every=25):
    model.eval()
    modules = dict(model.named_modules())
    summaries = []
    for stage, name in enumerate(targets, 1):
        bn = modules[name]
        moments = ChannelMoments()
        start = time.perf_counter()

        def capture(module, inputs):
            moments.update(inputs[0])
            # Already-calibrated upstream BNs stay in eval; downstream work is unnecessary.
            raise StopAtNorm()

        handle = bn.register_forward_pre_hook(capture)
        print(f"BN {stage}/{len(targets)}: {name}", flush=True)
        try:
            for i, images in enumerate(loader, 1):
                try:
                    model(images.to(device))
                except StopAtNorm:
                    pass
                else:
                    raise RuntimeError(f"Calibration target was not executed: {name}")
                if i == 1 or i % progress_every == 0 or i == len(loader):
                    print(f"  batches {i}/{len(loader)}", flush=True)
        finally:
            handle.remove()
        moments.apply(bn)
        summaries.append({"name": name, "observations_per_channel": moments.count,
                          "batches": moments.batches, "seconds": time.perf_counter() - start})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("Require positive samples/batch_size and nonnegative workers")
    source = args.checkpoint.resolve()
    output = args.output_dir.resolve() / "recovered.pth"
    if output.exists() or args.output_dir.exists():
        parser.error("Use a new output directory; original and previous results are never overwritten")
    if output == source:
        parser.error("Recovery output must not overwrite the source checkpoint")
    args.output_dir.mkdir(parents=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Source: {source}; device: {device}; only train/image is used", flush=True)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    model = WeavingUnet(pretrained_encoder=False).to(device)
    load_checkpoint_model(model, checkpoint)
    targets = split_mswd_stats(model)
    if len(targets) != 8:
        raise RuntimeError(f"Expected 8 independent MSWD BN sites, got {targets}")
    dataset = CalibrationCrops(args.data_root, args.samples, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    summaries = calibrate_layerwise(model, loader, targets, device)
    recovery = {"method": METHOD, "source_checkpoint": str(source), "source_epoch": checkpoint["epoch"],
                "calibration_split": "train", "labels_used": False, "weights_updated": False,
                "precision": "fp32", "samples": len(dataset), "seed": args.seed,
                "batch_size": args.batch_size, "targets": summaries, "crops": dataset.records}
    recovered = {key: checkpoint[key] for key in ("epoch", "args")}
    recovered.update({"model_state_dict": model.state_dict(), "bn_recovery": recovery,
                      "original_best_val_iou": checkpoint.get("best_val_iou"), "inference_only": True})
    temporary = output.with_suffix(".tmp")
    torch.save(recovered, temporary)
    temporary.replace(output)
    (args.output_dir / "calibration.json").write_text(json.dumps(recovery, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}; validate in normal eval before test", flush=True)


if __name__ == "__main__":
    main()
