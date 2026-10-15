"""Verify this experiment and index native Massachusetts validation/test masks."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from networks.skeleton_feature_architecture import BASE_COMMIT, require_checkpoint_architecture

NAME = "h0_skeleton_feature_sdgate_mass1500_random512_fp32_noema80_b2_acc1"
EXPECTED = dict(
    img_size=512, source_patch_size=1500, random_crop_train=True,
    random_crops_per_image=1, direct_resize_train=False, augment=True,
    batch_size=2, accumulation_steps=1, max_epochs=80, overlap_stride=256,
    amp_dtype="none", use_ema=False, structure_profile="stage23_boundary_0626",
    enable_highres_structure_stream=True, highres_structure_channels=64,
    highres_structure_fuse_stages="stage23", highres_structure_fusion_mode="stage23",
    enable_h3_surface_fusion=False, enable_post_refine_structure_interaction=False,
    remove_stage2_pre_topology_source=True, enable_global_topology=False,
    stage_skeleton_mode="direct", stage2_skeleton_weight=0.008,
    stage3_skeleton_weight=0.012, skeleton_pos_weight=2.0,
    stage_skeleton_loss_factor=1.0, stage_skeleton_only_loss_factor=0.5,
    stage2_skeleton_gradient_ratio=0.5, stage3_skeleton_gradient_ratio=0.5,
    stage3_gate_topology_gradient_ratio=0.0, highres_structure_skeleton_weight=0.01,
    stage_direction_factor=0.2, stage_connectivity_factor=3.0,
    road_attention_weight=0.0, surface_focal_gamma=1.0,
    masked_connectivity_center_experiment=True, connectivity_pos_weight=2.0,
    connectivity_focal_gamma=1.5, edge_contrastive_margin=0.1,
    directional_pos_weight_cardinal=1.0, directional_pos_weight_diagonal=2.5,
    pretrained_lr=5e-5, pretrained_min_lr=5e-6, new_lr=2e-4,
    new_min_lr=1e-5, warmup_epochs=10, threshold=0.2,
    skeleton_threshold=0.5, val_interval=5, trend_val_every_steps=0,
    max_train_batches=0,
)


def verify_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    meta = require_checkpoint_architecture(checkpoint)
    saved = checkpoint["args"]
    mismatches = {key: [saved.get(key), value] for key, value in EXPECTED.items()
                  if saved.get(key) != value}
    if mismatches or checkpoint.get("ema_state_dict") is not None:
        raise ValueError(f"Checkpoint differs from the requested experiment: {mismatches}")
    if not isinstance(saved.get("seed"), int):
        raise ValueError("Checkpoint must record the random seed")
    state = checkpoint["model_state_dict"]
    forbidden = ("swin_unet.global_topology.", "swin_unet.highres_structure_fusion.",
                 "swin_unet.surface_h3_fusion.", "swin_unet.structure_surface_correction_head.")
    if any(key.startswith(forbidden) or ".highres_skeleton_fusion." in key for key in state):
        raise ValueError("Checkpoint contains a removed H0 path")
    for stage in (2, 3):
        prefix = f"swin_unet.decoder_structure_blocks.{stage}."
        for key in ("skeleton_feature_interaction.alpha", "skeleton_feature_interaction.beta",
                    "skeleton_head.out.weight", "connectivity_head.edge_linear.weight"):
            if prefix + key not in state:
                raise ValueError(f"Missing stage {stage} parameter: {key}")
        if tuple(state[prefix + "connectivity_head.prior_embed.0.weight"].shape) != (16, 3, 1, 1):
            raise ValueError("C must read S_center/S_neighbor/D_alignment")
        if tuple(state[prefix + "reliability_correction.0.weight"].shape) != (48, 97, 3, 3):
            raise ValueError("Gate must contain the e2020d9 direction confidence correction")
        if prefix + "reliability_beta" not in state:
            raise ValueError("Gate reliability beta is missing")
    if tuple(state["swin_unet.surface_feature_delta.weight"].shape) != (96, 160, 1, 1):
        raise ValueError("Incorrect final surface fusion shape")
    return saved, checkpoint.get("epoch"), meta


def build_manifest(checkpoint, data_root, predictions_root, selection=None):
    saved, epoch, meta = verify_checkpoint(checkpoint)
    candidates = []
    for directory in sorted((predictions_root / "val").iterdir()):
        match = re.fullmatch(r"threshold_(\d+\.\d{2})", directory.name)
        if match and (directory / "surface").is_dir():
            candidates.append(dict(threshold=float(match.group(1)), path=str(directory / "surface")))
    if not candidates:
        raise ValueError("No exported validation threshold masks")
    test = []
    if selection:
        selected = json.loads(selection.read_text(encoding="utf-8"))
        if selected.get("selected_on") != "val" or NAME not in selected.get("models", {}):
            raise ValueError("Expected validation selection for this skeleton-feature experiment")
        threshold = float(selected["models"][NAME]["threshold"])
        directory = predictions_root / "test" / f"threshold_{threshold:.2f}" / "surface"
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        test = [dict(threshold=threshold, path=str(directory))]
    source_files = ["train_image.py", "test_image.py", "datasets/dataset_road_skeleton.py",
                    "networks/vision_transformer.py", "networks/skeleton_feature_architecture.py",
                    "networks/swin_transformer_unet_skip_expand_decoder_sys.py",
                    "networks/skeleton_guided_head.py", "losses/road_losses.py",
                    "evaluation/road_comparison/compare_predictions.py",
                    "evaluation/road_comparison/road_metrics.py"]
    provenance = dict(
        checkpoint=str(checkpoint), checkpoint_epoch=epoch,
        checkpoint_args_verified=True, base_model_commit=BASE_COMMIT,
        architecture=meta, seed=saved["seed"],
        physical_batch_size=2, accumulation_steps=1, effective_batch_size=2,
        training_crop="one native random512 crop per 1500 image per epoch; shared epoch",
        training_precision="FP32", training_max_epochs=80, weights="model_state_dict",
        ema=False, inference_tile_size=512, overlap_stride=256,
        merge="taper-weighted logits (1D weight floor0.05), divisor clamp1e-8, sigmoid",
        tta="none", postprocessing="surface threshold only",
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in source_files},
    )
    return dict(data_root=str(data_root),
                protocol=dict(image_size=1500, short_area_threshold=20,
                              apls_max_nodes=64, apls_snap_radius=5.0),
                models=[dict(name=NAME, provenance=provenance,
                             predictions=dict(val=candidates, test=test))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--predictions_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        _, epoch, _ = verify_checkpoint(args.checkpoint)
        print(f"Verified {NAME}, saved epoch={epoch}", flush=True)
        return
    manifest = build_manifest(args.checkpoint, args.data_root, args.predictions_root, args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
