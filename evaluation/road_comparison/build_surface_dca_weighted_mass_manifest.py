"""Verify the e2020d9 Massachusetts protocol and index exported binary masks."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch


BASE = "e2020d9a8df311f9f680cd3637466a674d76688e"
NAME = "e2020d9_surface_dca_weighted_mass1500_random512_fp32_noema80_b2_acc1"
EXPECTED = dict(
    surface_only=True,
    img_size=512, source_patch_size=1500, random_crop_train=True,
    random_crops_per_image=1, direct_resize_train=False, augment=True,
    batch_size=2, accumulation_steps=1, max_epochs=80, overlap_stride=256,
    amp_dtype="none", use_ema=False, structure_profile="stage23_boundary_0626",
    enable_highres_structure_stream=True, highres_structure_channels=64,
    highres_structure_fuse_stages="stage23", highres_structure_fusion_mode="stage23",
    enable_e128_stage_fusion=True, enable_h3_surface_fusion=True,
    enable_post_refine_structure_interaction=False,
    remove_stage2_pre_topology_source=True, enable_global_topology=True,
    global_topology_max_nodes=32, global_topology_heads=4, global_topology_alpha_max=0.05,
    stage_skeleton_mode="direct", stage2_skeleton_weight=0.008,
    stage3_skeleton_weight=0.012, skeleton_pos_weight=2.0,
    stage_skeleton_loss_factor=1.0, stage_skeleton_only_loss_factor=1.0,
    stage2_skeleton_gradient_ratio=0.5, stage3_skeleton_gradient_ratio=0.5,
    stage3_gate_topology_gradient_ratio=0.0, highres_structure_skeleton_weight=0.0,
    stage_direction_factor=0.2, stage_connectivity_factor=3.0,
    road_attention_weight=0.0, surface_focal_gamma=1.0,
    masked_connectivity_center_experiment=True, connectivity_pos_weight=2.0,
    connectivity_focal_gamma=1.5, edge_contrastive_margin=0.1,
    directional_pos_weight_cardinal=1.0, directional_pos_weight_diagonal=2.5,
    base_lr=5e-4, min_lr=1e-5, pretrained_lr=5e-5, pretrained_min_lr=5e-6,
    new_lr=2e-4, new_min_lr=1e-5, warmup_epochs=10,
    threshold=0.2, skeleton_threshold=0.5, val_interval=5,
    trend_val_every_steps=0, max_train_batches=0,
)


def verify_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("args", {})
    mismatches = {key: [saved.get(key), value] for key, value in EXPECTED.items()
                  if saved.get(key) != value}
    if mismatches or checkpoint.get("ema_state_dict") is not None:
        raise ValueError(f"Checkpoint differs from this Mass protocol: {mismatches}")
    if not isinstance(saved.get("seed"), int):
        raise ValueError("Checkpoint must record the training seed")
    if not checkpoint.get("model_state_dict"):
        raise ValueError("Checkpoint has no model weights")
    for index in (0, 1):
        key = f"swin_unet.dca_blocks.{index}.weighted_fusion_version"
        version = checkpoint["model_state_dict"].get(key)
        if not isinstance(version, torch.Tensor) or version.numel() != 1 or version.item() != 1:
            raise ValueError("Expected a newly trained weighted-DCA checkpoint, not the old surface-only run")
    return saved, checkpoint.get("epoch")


def build_manifest(checkpoint, data_root, predictions_root, selection=None):
    saved, epoch = verify_checkpoint(checkpoint)
    candidates = []
    for directory in sorted((predictions_root / "val").iterdir()):
        match = re.fullmatch(r"threshold_(\d+\.\d{2})", directory.name)
        if match and (directory / "surface").is_dir():
            candidates.append(dict(threshold=float(match.group(1)), path=str(directory / "surface")))
    if not candidates:
        raise ValueError("No exported validation masks")
    test = []
    if selection is not None:
        chosen = json.loads(selection.read_text(encoding="utf-8"))
        if chosen.get("selected_on") != "val" or NAME not in chosen.get("models", {}):
            raise ValueError("Expected a validation selection for this model")
        threshold = float(chosen["models"][NAME]["threshold"])
        directory = predictions_root / "test" / f"threshold_{threshold:.2f}" / "surface"
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        test = [dict(threshold=threshold, path=str(directory))]
    root = Path(__file__).resolve().parents[2]
    files = ["train_image.py", "test_image.py", "datasets/dataset_road_skeleton.py",
             "config.py", "configs/swin_tiny_patch4_window7_224_lite.yaml",
             "networks/vision_transformer.py", "networks/surface_only.py", "networks/dca_fpn_lite.py",
             "networks/swin_transformer_unet_skip_expand_decoder_sys.py",
             "networks/skeleton_guided_head.py", "networks/keypoint_global_topology.py",
             "losses/road_losses.py", "losses/cldice_loss.py", "direction_target_utils.py",
             "evaluation/road_comparison/compare_predictions.py",
             "evaluation/road_comparison/road_metrics.py"]
    return dict(
        data_root=str(data_root),
        protocol=dict(image_size=1500, short_area_threshold=20,
                      apls_max_nodes=64, apls_snap_radius=5.0),
        models=[dict(name=NAME, provenance=dict(
            checkpoint=str(checkpoint), checkpoint_epoch=epoch,
            base_model_commit=BASE, checkpoint_args_verified=True,
            surface_only_base_commit="eb2da24d768b7f643e4fb288e3736693862c1fe8",
            dca_fusion="scaled_softmax_weighted_concat_v1",
            dca_weight_initialization="zero final weight predictor; K * softmax = 1 at initialization",
            seed=saved["seed"], pretrained_checkpoint=saved.get("pretrain_ckpt"),
            loaded_pretrained_names_record="checkpoint.loaded_pretrained_names",
            training_crop="one random512 crop per native1500 image each epoch",
            crop_schedule="original e2020d9 epoch-seeded RNG; shared worker epoch",
            skeleton_target="none; surface-only ablation",
            ablation="remove A1/A2, S/C/D, highres/E128, structure feedback, boundary and global topology",
            physical_batch_size=2, accumulation_steps=1, training_max_epochs=80,
            training_precision="FP32", weights="model_state_dict", ema=False,
            inference_tile_size=512, overlap_stride=256,
            merge="taper0.1-weighted logits, divisor clamp1e-8, sigmoid",
            tta="none", postprocessing="surface threshold only",
            source_sha256={path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                           for path in files}), predictions=dict(val=candidates, test=test))],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--predictions_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        _, epoch = verify_checkpoint(args.checkpoint)
        print(f"Verified {NAME}, epoch={epoch}", flush=True)
        return
    manifest = build_manifest(args.checkpoint, args.data_root, args.predictions_root, args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
