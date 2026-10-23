"""Three additional frozen road probes for the 1af9580 encoder checkpoint.

E1 and H0 come from the encoder feature cache. F1 is the output of decoder
layers_up[1], after its 1/16 skip and upsampling to the 1/8 grid. No backbone
parameter is trained. The existing seven probe heads remain untouched.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


EXTRA_NAMES = ("E1", "H0", "F1")
PROTOCOL = "h0-extra-three-road-probes-v1"


class ExtraFeatureTap:
    def __init__(self, base_tap):
        self.base_tap = base_tap
        self.model = base_tap.model
        self.f1 = None
        self.handle = self.model.swin_unet.layers_up[1].register_forward_hook(self._f1_hook)

    def _f1_hook(self, _module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise RuntimeError("F1 decoder output must be B,L,C tokens")
        batch, tokens, channels = output.shape
        width = math.isqrt(tokens)
        if width * width != tokens:
            raise RuntimeError(f"F1 token count {tokens} is not a square grid")
        self.f1 = output.detach().transpose(1, 2).reshape(batch, channels, width, width)

    def __call__(self, images):
        self.f1 = None
        features, original = self.base_tap(images)
        encoder = self.model.swin_unet.last_encoder_features
        if self.f1 is None or not all(name in encoder for name in ("E1", "H0")):
            raise RuntimeError("Missing E1/H0/F1 feature tap")
        extra = {"E1": encoder["E1"].detach(), "H0": encoder["H0"].detach(), "F1": self.f1}
        if any(value.ndim != 4 or value.shape[0] != images.shape[0] for value in extra.values()):
            raise RuntimeError("Invalid extra probe feature shape")
        return {**features, **extra}, original

    def close(self):
        self.handle.remove()
        self.base_tap.close()


def make_heads(channels, device):
    heads = nn.ModuleDict({name: nn.Conv2d(channels[name], 1, 1) for name in EXTRA_NAMES}).to(device)
    for head in heads.values():
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    return heads


def load_extra_heads(path, backbone_sha256, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != PROTOCOL or payload.get("backbone_sha256") != backbone_sha256:
        raise ValueError("Extra probe checkpoint does not match the encoder backbone")
    if tuple(payload.get("probe_names", ())) != EXTRA_NAMES:
        raise ValueError("Extra probe names differ from this code")
    heads = make_heads(payload["channels"], device)
    for name in EXTRA_NAMES:
        heads[name].load_state_dict(payload["best_states"][name], strict=True)
    return heads.eval(), payload


@torch.no_grad()
def validate(exp, tap, heads, loader, device):
    totals = {name: np.zeros(3, dtype=np.int64) for name in EXTRA_NAMES}
    losses = {name: 0.0 for name in EXTRA_NAMES}
    count = 0
    for images, target in loader:
        images, target = images.to(device), target.to(device)
        features, _ = tap(images)
        count += images.shape[0]
        for name in EXTRA_NAMES:
            logits = exp.probe_logits(heads[name], features[name])
            losses[name] += float(exp.road_loss(logits, target)) * images.shape[0]
            pred = logits.sigmoid() >= .5
            actual = target >= .5
            totals[name] += np.array([(pred & actual).sum().item(),
                                      (pred & ~actual).sum().item(),
                                      (~pred & actual).sum().item()], dtype=np.int64)
    if count == 0:
        raise ValueError("Empty validation set")
    return {name: {**exp.scores(*totals[name]), "val_loss": losses[name] / count}
            for name in EXTRA_NAMES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe_repo", required=True)
    parser.add_argument("--probe_model_path", required=True)
    parser.add_argument("--probe_head_path", required=True, help="Existing seven-probe checkpoint")
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_train_images", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Choose an empty output directory")
    if args.max_epochs < 1 or args.batch_size < 1:
        parser.error("max_epochs and batch_size must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    # The loader imports the 1af architecture from --probe_repo and checks the
    # original seven-probe checkpoint against the same exact backbone bytes.
    from diagnose_gap_encoder_evidence import load_probe_stack
    device = torch.device(args.device)
    exp, model, _, base_tap, old_payload = load_probe_stack(
        Path(args.probe_repo), Path(args.probe_model_path), Path(args.probe_head_path), device)
    tap = ExtraFeatureTap(base_tap)
    train_data = exp.RoadOnlyCrop(args.root_path, "train", args.seed, args.max_train_images)
    val_data = exp.RoadOnlyCrop(args.root_path, "val", args.seed, 0)
    worker_count = args.num_workers
    generator = torch.Generator()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=worker_count, persistent_workers=worker_count > 0,
                              pin_memory=True, generator=generator)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=worker_count, persistent_workers=worker_count > 0,
                            pin_memory=True)
    try:
        first, _ = train_data[0]
        features, _ = tap(first[None].to(device))
        channels = {name: int(features[name].shape[1]) for name in EXTRA_NAMES}
        shapes = {name: list(features[name].shape[1:]) for name in EXTRA_NAMES}
        if shapes["E1"][-2:] != [128, 128] or shapes["H0"][-2:] != [128, 128] or shapes["F1"][-2:] != [64, 64]:
            raise RuntimeError(f"Unexpected E1/H0/F1 grids: {shapes}")
        print(f"Frozen extra probe shapes: {shapes}", flush=True)
        heads = make_heads(channels, device)
        optimizer = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=1e-4)
        if {id(p) for p in model.parameters()} & {id(p) for p in heads.parameters()}:
            raise RuntimeError("Backbone parameter entered probe optimizer")
        best_iou = {name: -1.0 for name in EXTRA_NAMES}
        best_loss = {name: float("inf") for name in EXTRA_NAMES}
        best_states = {}
        best_epoch = {}
        history = []
        stale = 0
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, args.max_epochs + 1):
            train_data.set_epoch(epoch - 1)
            generator.manual_seed(args.seed + epoch)
            heads.train()
            running, batches = 0.0, 0
            for images, target in train_loader:
                images, target = images.to(device), target.to(device)
                features, _ = tap(images)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.stack([exp.road_loss(exp.probe_logits(heads[name], features[name]), target)
                                    for name in EXTRA_NAMES]).mean()
                loss.backward()
                optimizer.step()
                running += float(loss.detach()); batches += 1
            heads.eval()
            result = validate(exp, tap, heads, val_loader, device)
            improved = False
            for name in EXTRA_NAMES:
                metric = result[name]
                if metric["iou"] > best_iou[name] + 1e-6 or (abs(metric["iou"] - best_iou[name]) <= 1e-6
                        and metric["val_loss"] < best_loss[name] - 1e-5):
                    best_iou[name], best_loss[name] = metric["iou"], metric["val_loss"]
                    best_states[name] = copy.deepcopy({key: value.detach().cpu() for key, value in heads[name].state_dict().items()})
                    best_epoch[name] = epoch
                    improved = True
            history.append({"epoch": epoch, "train_loss": running / max(batches, 1), "validation": result})
            payload = {"protocol": PROTOCOL, "backbone_sha256": old_payload["backbone_sha256"],
                       "architecture_id": old_payload["architecture_id"], "probe_names": EXTRA_NAMES,
                       "channels": channels, "feature_shapes": shapes, "best_states": best_states,
                       "best_iou": best_iou, "best_epoch": best_epoch, "seed": args.seed,
                       "train_image_count": len(train_data), "val_image_count": len(val_data),
                       "selection": "best per-head validation center crop IoU at threshold 0.5",
                       "training": "frozen backbone; one random512 crop per train image per epoch; only three 1x1 heads optimized"}
            exp.atomic_save(payload, args.output_dir / "extra_probe_best.pth")
            (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"[EXTRA-PROBES] epoch={epoch} train_loss={running/max(batches,1):.5f} "
                  f"val_iou={[round(result[n]['iou'], 5) for n in EXTRA_NAMES]}", flush=True)
            stale = 0 if improved else stale + 1
            if epoch >= 5 and stale >= args.patience:
                break
    finally:
        tap.close()


if __name__ == "__main__":
    main()
