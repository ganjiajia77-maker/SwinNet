import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_road_skeleton import RoadSkeletonDataset
from config import get_config
from networks.vision_transformer_selective_fusion import (
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
)


REGIONS = ("WeakFN", "SkeletonTP", "HardBG")
EXPECTED_COUNTS = {"WeakFN": 7783, "SkeletonTP": 32204, "HardBG": 103760}


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collect_batches(loader, max_batches):
    batches = []
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batches.append(batch)
    return batches


def resize_mask(mask, spatial_size):
    if mask.shape[-2:] != spatial_size:
        mask = F.interpolate(mask.float(), size=spatial_size, mode="nearest")
    return mask


def run_normal_alpha1(model, swin, images):
    outputs = model(images)
    return outputs


def build_model(args, checkpoint):
    config = get_config(args)
    model = ViT_seg(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        return_skeleton=True,
        final_topology_eta_init=args.final_topology_eta_init,
        final_gap_rho_init=args.final_gap_rho_init,
        stage_topology_stages=args.stage_topology_stages,
        stage_topology_alpha_max=args.stage_topology_alpha_max,
        stage_topology_alpha_init=args.stage_topology_alpha_init,
        structure_profile=args.structure_profile,
        use_msfe_skip=not args.disable_msfe_skip,
        enable_highres_structure_stream=args.enable_highres_structure_stream,
        highres_structure_channels=args.highres_structure_channels,
        highres_structure_fuse_stages=args.highres_structure_fuse_stages,
        highres_structure_fusion_mode=args.highres_structure_fusion_mode,
        enable_post_refine_structure_interaction=(
            args.enable_post_refine_structure_interaction
        ),
    )
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=False,
    )
    return model.to(args.device).eval()


def apply_checkpoint_args(args, checkpoint):
    if not isinstance(checkpoint.get("args"), dict):
        return
    saved_args = checkpoint["args"]
    args.structure_profile = saved_args.get("structure_profile", args.structure_profile)
    args.disable_msfe_skip = bool(saved_args.get("disable_msfe_skip", args.disable_msfe_skip))
    args.enable_highres_structure_stream = bool(saved_args.get("enable_highres_structure_stream", args.enable_highres_structure_stream))
    args.highres_structure_channels = int(saved_args.get("highres_structure_channels", args.highres_structure_channels))
    args.highres_structure_fuse_stages = str(saved_args.get("highres_structure_fuse_stages", args.highres_structure_fuse_stages))
    args.highres_structure_fusion_mode = str(saved_args.get("highres_structure_fusion_mode", args.highres_structure_fusion_mode))
    args.enable_post_refine_structure_interaction = bool(
        saved_args.get(
            "enable_post_refine_structure_interaction",
            args.enable_post_refine_structure_interaction,
        )
        or args.enable_post_refine_structure_interaction
    )


def append_region_features(store, feature_map, masks):
    if feature_map.shape[-2:] != next(iter(masks.values())).shape[-2:]:
        feature_map = F.interpolate(
            feature_map,
            size=next(iter(masks.values())).shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    feature_map = feature_map.detach().cpu()
    for name, mask in masks.items():
        values = feature_map.permute(0, 2, 3, 1)[mask]
        if values.numel() > 0:
            store[name].append(values.float())


def resize_feature(feature_map, target_hw):
    if feature_map.shape[-2:] != target_hw:
        feature_map = F.interpolate(
            feature_map,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
    return feature_map


def extract_z_struct(args, swin, images):
    z_struct = getattr(swin, "last_highres_z_struct", None)
    if z_struct is not None:
        return resize_feature(z_struct.detach().cpu(), images.shape[-2:])

    encoder = None
    if hasattr(swin, "prepatch_structure_encoder") and swin.prepatch_structure_encoder is not None:
        encoder = swin.prepatch_structure_encoder
    elif hasattr(swin, "highres_structure_encoder") and swin.highres_structure_encoder is not None:
        encoder = swin.highres_structure_encoder
    if encoder is None:
        raise RuntimeError(
            "Cannot extract Z_struct: checkpoint/model has no highres/prepatch structure encoder."
        )
    return resize_feature(encoder(images).detach().cpu(), images.shape[-2:])


def collect_decoder_structure_features(outputs, target_hw):
    if not isinstance(outputs, tuple) or len(outputs) < 2:
        return None, None
    structure_outputs = outputs[-1]
    if not isinstance(structure_outputs, list):
        return None, None

    connectivity_maps = []
    direction_maps = []
    for item in structure_outputs:
        if not isinstance(item, dict):
            continue
        if "connectivity" not in item or "direction" not in item:
            continue
        stage = item.get("stage")
        if not isinstance(stage, int):
            continue
        connectivity_maps.append(resize_feature(item["connectivity"], target_hw))
        direction_maps.append(resize_feature(item["direction"], target_hw))

    if not connectivity_maps or not direction_maps:
        return None, None
    connectivity = torch.cat(connectivity_maps, dim=1)
    direction = torch.cat(direction_maps, dim=1)
    return connectivity, direction


def get_stage3_decoder_block(swin):
    if not hasattr(swin, "decoder_structure_blocks"):
        raise RuntimeError("Model has no decoder_structure_blocks.")
    if len(swin.decoder_structure_blocks) < 4:
        raise RuntimeError("decoder_structure_blocks does not include stage 3.")
    return swin.decoder_structure_blocks[3]


def attach_stage3_feature_hooks(swin):
    block = get_stage3_decoder_block(swin)

    def save_hidden(_module, _inp, out):
        block._probe_hidden = out.detach().cpu()

    def save_connectivity(_module, _inp, out):
        block._probe_connectivity_logits = out.detach().cpu()

    def save_direction(_module, _inp, out):
        block._probe_direction_logits = out.detach().cpu()

    block._probe_hidden = None
    block._probe_connectivity_logits = None
    block._probe_direction_logits = None
    block.structure_branch.register_forward_hook(save_hidden)
    block.connectivity_head.register_forward_hook(save_connectivity)
    block.direction_head.register_forward_hook(save_direction)
    block._probe_source = "decoder_structure_blocks[3]"
    return block


def collect_feature_sets(args, batches, model, swin):
    feature_sets = {
        "z_struct": {name: [] for name in REGIONS},
        "connectivity": {name: [] for name in REGIONS},
        "con_hidden": {name: [] for name in REGIONS},
        "con_logits": {name: [] for name in REGIONS},
        "con_mean_prob": {name: [] for name in REGIONS},
        "con_max_prob": {name: [] for name in REGIONS},
        "direction": {name: [] for name in REGIONS},
        "z_struct+connectivity": {name: [] for name in REGIONS},
        "z_struct+con_hidden": {name: [] for name in REGIONS},
        "z_struct+con_logits": {name: [] for name in REGIONS},
        "z_struct+con_mean_prob": {name: [] for name in REGIONS},
        "z_struct+con_max_prob": {name: [] for name in REGIONS},
        "z_struct+direction": {name: [] for name in REGIONS},
        "z_struct+connectivity+direction": {name: [] for name in REGIONS},
    }
    counts = {name: 0 for name in REGIONS}
    stage3_block = attach_stage3_feature_hooks(swin)
    last_z_struct = None

    with torch.no_grad():
        for batch in tqdm(batches, desc="Collect frozen features"):
            images = batch["image"].to(args.device)
            outputs = run_normal_alpha1(model, swin, images)
            if not isinstance(outputs, tuple):
                raise RuntimeError("Expected tuple outputs from structure-guided model.")
            logits = outputs[0].detach().cpu()
            prob = torch.sigmoid(logits).squeeze(1)
            surface_gt = resize_mask(batch["mask"].float(), logits.shape[-2:]).squeeze(1) > 0.5
            skeleton = resize_mask(batch["skeleton"].float(), logits.shape[-2:]).squeeze(1) > 0.5
            masks = {
                "WeakFN": skeleton & (prob < args.threshold),
                "SkeletonTP": skeleton & (prob >= args.threshold),
                "HardBG": (~surface_gt) & (prob >= args.threshold),
            }
            for name in REGIONS:
                counts[name] += int(masks[name].sum().item())

            z_struct = extract_z_struct(args, swin, images)
            last_z_struct = z_struct
            connectivity, direction = collect_decoder_structure_features(
                outputs,
                logits.shape[-2:],
            )
            if connectivity is None or direction is None:
                raise RuntimeError(
                    "Could not extract decoder structure connectivity/direction outputs from model forward."
                )
            connectivity = connectivity.detach().cpu()
            direction = direction.detach().cpu()
            con_hidden = stage3_block._probe_hidden
            con_logits = stage3_block._probe_connectivity_logits
            if con_hidden is None or con_logits is None:
                raise RuntimeError("Stage3 feature hooks did not capture hidden/logits.")
            con_hidden = resize_feature(con_hidden, logits.shape[-2:])
            con_logits = resize_feature(con_logits, logits.shape[-2:])
            con_prob = torch.sigmoid(con_logits)
            con_mean_prob = con_prob.mean(dim=1, keepdim=True)
            con_max_prob = con_prob.max(dim=1, keepdim=True).values

            append_region_features(feature_sets["z_struct"], z_struct, masks)
            append_region_features(feature_sets["connectivity"], connectivity, masks)
            append_region_features(feature_sets["con_hidden"], con_hidden, masks)
            append_region_features(feature_sets["con_logits"], con_logits, masks)
            append_region_features(feature_sets["con_mean_prob"], con_mean_prob, masks)
            append_region_features(feature_sets["con_max_prob"], con_max_prob, masks)
            append_region_features(feature_sets["direction"], direction, masks)
            append_region_features(
                feature_sets["z_struct+connectivity"],
                torch.cat([z_struct, connectivity], dim=1),
                masks,
            )
            append_region_features(
                feature_sets["z_struct+con_hidden"],
                torch.cat([z_struct, con_hidden], dim=1),
                masks,
            )
            append_region_features(
                feature_sets["z_struct+con_logits"],
                torch.cat([z_struct, con_logits], dim=1),
                masks,
            )
            append_region_features(
                feature_sets["z_struct+con_mean_prob"],
                torch.cat([z_struct, con_mean_prob], dim=1),
                masks,
            )
            append_region_features(
                feature_sets["z_struct+con_max_prob"],
                torch.cat([z_struct, con_max_prob], dim=1),
                masks,
            )
            append_region_features(
                feature_sets["z_struct+direction"],
                torch.cat([z_struct, direction], dim=1),
                masks,
            )
            append_region_features(
                feature_sets["z_struct+connectivity+direction"],
                torch.cat([z_struct, connectivity, direction], dim=1),
                masks,
            )

    return {
        feature_name: {
            region: torch.cat(chunks, dim=0).numpy() if chunks else np.empty((0, 0), dtype=np.float32)
            for region, chunks in regions.items()
        }
        for feature_name, regions in feature_sets.items()
    }, counts


def balanced_probe(features, seed, test_size):
    positive = features["WeakFN"]
    negative = features["HardBG"]
    sample_count = min(len(positive), len(negative))
    rng = np.random.default_rng(seed)
    pos_idx = rng.choice(len(positive), size=sample_count, replace=False)
    neg_idx = rng.choice(len(negative), size=sample_count, replace=False)

    x = np.concatenate([positive[pos_idx], negative[neg_idx]], axis=0)
    y = np.concatenate(
        [
            np.ones(sample_count, dtype=np.int64),
            np.zeros(sample_count, dtype=np.int64),
        ],
        axis=0,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
        ),
    )
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        pred,
        average="binary",
        zero_division=0,
    )
    return {
        "balanced_samples_per_class": sample_count,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "auc": float(roc_auc_score(y_test, prob)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=39)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)
    parser.add_argument("--final_gap_rho_init", type=float, default=0.005)
    parser.add_argument("--stage_topology_stages", type=str, default="none", choices=["none", "stage3", "stage23"])
    parser.add_argument("--stage_topology_alpha_max", type=float, default=1.0)
    parser.add_argument("--stage_topology_alpha_init", type=float, default=0.1)
    parser.add_argument("--cfg", type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache_mode", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--accumulation_steps", type=int, default=0)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--amp_opt_level", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--dataset", type=str, default="ImageData")
    parser.add_argument("--n_class", default=2, type=int)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--structure_profile", type=str, default="full")
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23", choices=["stage2", "stage3", "stage23"])
    parser.add_argument(
        "--highres_structure_fusion_mode",
        type=str,
        default="stage23",
        choices=[
            "stage23",
            "final_correction",
            "stage23_final_correction",
            "post_refine_interaction",
            "none",
        ],
    )
    parser.add_argument("--enable_post_refine_structure_interaction", action="store_true")
    args = parser.parse_args()

    set_deterministic(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model_path, map_location="cpu")
    apply_checkpoint_args(args, checkpoint)
    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    batches = collect_batches(loader, args.max_batches)
    model = build_model(args, checkpoint)
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet

    feature_sets, counts = collect_feature_sets(args, batches, model, swin)
    print(f"\nCheckpoint: {args.model_path}")
    print(f"split={args.split}, batches={len(batches)}, images={sum(batch['image'].shape[0] for batch in batches)}")
    print(f"threshold={args.threshold}, seed={args.seed}, test_size={args.test_size}")
    print("Masks from alpha=1 normal:")
    print("  WeakFN = GT skeleton & sigmoid(logits_normal) < threshold")
    print("  SkeletonTP = GT skeleton & sigmoid(logits_normal) >= threshold")
    print("  HardBG = surface_gt == 0 & sigmoid(logits_normal) >= threshold")
    print("\nPixel counts:")
    for name in REGIONS:
        print(f"  {name}: {counts[name]} expected={EXPECTED_COUNTS[name]} match={counts[name] == EXPECTED_COUNTS[name]}")

    print("\nFeature shapes by region:")
    for feature_name, regions in feature_sets.items():
        parts = [f"{region}={regions[region].shape}" for region in REGIONS]
        print(f"  {feature_name}: " + ", ".join(parts))
    z_shape = feature_sets["z_struct"]["WeakFN"].shape[1:]
    con_hidden_shape = feature_sets["con_hidden"]["WeakFN"].shape[1:]
    con_logits_shape = feature_sets["con_logits"]["WeakFN"].shape[1:]
    dir_shape = feature_sets["direction"]["WeakFN"].shape[1:]
    print(
        "\nFeature sources:"
        f"\n  Z_struct source = highres/prepatch structure encoder, shape = {z_shape}"
        f"\n  Con hidden source = stage3 decoder block structure_branch, shape = {con_hidden_shape}"
        f"\n  Con logits source = stage3 decoder block connectivity_head, shape = {con_logits_shape}"
        f"\n  Dir source = stage3 decoder block direction_head, shape = {dir_shape}"
    )

    print("\n========== Balanced linear probe: WeakFN(class 1) vs HardBG(class 0) ==========")
    print(f"{'feature':<14} {'n/class':>8} {'train':>8} {'test':>8} {'AUC':>8} {'P':>8} {'R':>8} {'F1':>8}")
    for feature_name in (
        "z_struct",
        "con_hidden",
        "con_logits",
        "con_mean_prob",
        "con_max_prob",
        "connectivity",
        "direction",
        "z_struct+connectivity",
        "z_struct+con_hidden",
        "z_struct+con_logits",
        "z_struct+con_mean_prob",
        "z_struct+con_max_prob",
        "z_struct+direction",
        "z_struct+connectivity+direction",
    ):
        result = balanced_probe(feature_sets[feature_name], args.seed, args.test_size)
        print(
            f"{feature_name:<14} {result['balanced_samples_per_class']:>8d} "
            f"{result['train_samples']:>8d} {result['test_samples']:>8d} "
            f"{result['auc']:>8.4f} {result['precision']:>8.4f} "
            f"{result['recall']:>8.4f} {result['f1']:>8.4f}"
        )


if __name__ == "__main__":
    main()
