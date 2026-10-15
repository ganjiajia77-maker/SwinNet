"""Implementation checks only: shapes, gradients, loss accounting and strict reload."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import SurfaceStructureLoss, build_soft_stage_skeleton_target
from networks.skeleton_feature_architecture import (
    ARCHITECTURE_ID, architecture_metadata, require_checkpoint_architecture, restore_saved_config, restore_model_config,
)
from networks.vision_transformer import SwinUnet


def model_args():
    return SimpleNamespace(
        cfg=str(ROOT / "configs/swin_tiny_patch4_window7_224_lite.yaml"), opts=None,
        batch_size=2, img_size=512, zip=False, cache_mode="", resume="",
        accumulation_steps=1, use_checkpoint=False, amp_opt_level="", tag="",
        eval=False, throughput=False, source_patch_size=1500,
        enable_highres_structure_stream=True, highres_structure_channels=64,
        stage_skeleton_mode="direct", enable_global_topology=False,
        stage2_skeleton_weight=0.008, stage3_skeleton_weight=0.012,
        stage_skeleton_loss_factor=1.0, stage_skeleton_only_loss_factor=0.5,
        highres_structure_skeleton_weight=0.01,
    )


def make_model(args):
    return SwinUnet(
        get_config(args), img_size=512, num_classes=1, return_skeleton=True,
        structure_profile="stage23_boundary_0626", stage_skeleton_mode="direct",
        enable_highres_structure_stream=True, highres_structure_channels=64,
        remove_stage2_pre_topology_source=True, enable_global_topology=False,
    )


def grad_norm(module):
    return sum(float(p.grad.detach().abs().sum()) for p in module.parameters() if p.grad is not None)


def check_crop_workers():
    with tempfile.TemporaryDirectory(prefix="skeleton_crop_check_") as tmp:
        root = Path(tmp)
        (root / "train/image").mkdir(parents=True)
        (root / "train/mask").mkdir(parents=True)
        y, x = np.indices((1500, 1500))
        image = np.stack([x % 256, y % 256, (x + y) % 256], axis=-1).astype(np.uint8)
        mask = np.zeros((1500, 1500), np.uint8)
        cv2.line(mask, (0, 700), (1499, 700), 255, 7)
        assert cv2.imwrite(str(root / "train/image/example.tiff"), image)
        assert cv2.imwrite(str(root / "train/mask/example.tiff"), mask)
        dataset = RoadSkeletonDataset(str(root), image_size=512, source_patch_size=1500,
                                      tile_size=512, random_crop_train=True, augment=False)
        loader = DataLoader(dataset, batch_size=1, num_workers=2, persistent_workers=True)
        coords = []
        for epoch in (0, 1, 2, 0):
            dataset.set_epoch(epoch)
            batch = next(iter(loader))
            coords.append([int(batch["tile_top"]), int(batch["tile_left"])])
            assert tuple(batch["image"].shape) == (1, 3, 512, 512)
        assert coords[0] == coords[3] and len({tuple(pair) for pair in coords[:3]}) == 3
        del batch, loader, dataset
    return coords


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "skeleton_feature_validation.json")
    args = parser.parse_args()
    torch.manual_seed(1234)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = model_args()
    model = make_model(cfg).to(device).eval()
    core = model.swin_unet
    assert core.global_topology is None
    assert not hasattr(core, "_surface_prior_for_global_topology")
    assert not any(type(m).__name__ in {"KeypointGuidedGlobalTopology", "HighResStructureFusion",
                                      "BidirectionalSkeletonFeatureFusion"} for m in model.modules())
    assert torch.count_nonzero(core.surface_feature_delta.weight) == 0
    assert torch.count_nonzero(core.surface_feature_delta.bias) == 0
    report = dict(architecture_id=ARCHITECTURE_ID, device=str(device), torch_version=torch.__version__,
                  parameters=sum(p.numel() for p in model.parameters()))
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(torch.randn(1, 3, 512, 512, device=device))
    report["forward_flops_batch1_512"] = counter.get_total_flops()
    print("[CHECK] parameters/FLOPs", report, flush=True)
    calls = {"guided_head": 0, "surface_head": 0}
    def count(name):
        def hook(*_):
            calls[name] += 1
        return hook
    hooks = [core.guided_head.register_forward_hook(count("guided_head")),
             core.guided_head.surface_head.register_forward_hook(count("surface_head"))]
    model.train()
    x = torch.randn(2, 3, 512, 512, device=device)
    outputs = model(x)
    assert calls == {"guided_head": 1, "surface_head": 1}, calls
    for hook in hooks:
        hook.remove()
    report["surface_forward_calls_per_model_call"] = calls
    expected = {"H0": (2, 64, 128, 128), "H1": (2, 64, 128, 128),
                "H2": (2, 64, 128, 128), "G2": (2, 96, 128, 128),
                "G3": (2, 96, 128, 128), "B2": (2, 96, 128, 128),
                "B3": (2, 96, 128, 128)}
    report["feature_shapes"] = {key: list(core.last_stage_features[key].shape) for key in expected}
    assert {key: tuple(value) for key, value in report["feature_shapes"].items()} == expected
    stage_outputs = outputs[4]
    stage_heads = [item for item in stage_outputs if item.get("stage") in (2, 3)]
    assert [item["stage"] for item in stage_heads] == [2, 3]
    assert all(tuple(item["skeleton"].shape) == (2, 1, 128, 128) for item in stage_heads)
    assert tuple(core.last_highres_structure_skeleton.shape) == (2, 1, 128, 128)
    assert sum("highres_structure_skeleton" in item for item in stage_outputs) == 1
    assert outputs[0].shape == (2, 1, 512, 512)
    criterion = SurfaceStructureLoss(
        skeleton_weight=0, connectivity_weight=0, stage_structure_weights=(0, 0, .008, .012),
        stage_skeleton_only_loss_factor=.5, highres_structure_skeleton_weight=.01,
        stage_direction_factor=.2, stage_connectivity_factor=3,
        use_masked_connectivity_center_experiment=True, skeleton_pos_weight=2,
        connectivity_pos_weight=2, connectivity_focal_gamma=1.5,
        directional_pos_weight_cardinal=1, directional_pos_weight_diagonal=2.5,
        edge_contrastive_margin=.1, surface_focal_gamma=1,
    ).to(device)
    skeleton = torch.zeros(2, 1, 512, 512, device=device)
    skeleton[:, :, 256, 30:480] = 1
    skeleton[:, :, 30:480, 240] = 1
    dilate = torch.nn.functional.max_pool2d(skeleton, 3, stride=1, padding=1)
    surface_target = torch.nn.functional.max_pool2d(skeleton, 9, stride=1, padding=4)
    sh_loss, sh_stats = criterion.highres_structure_skeleton_loss(stage_outputs, skeleton, dilate)
    sh_loss.backward(retain_graph=True)
    report["SH_gradient_abs_sum"] = {
        "encoder": grad_norm(core.prepatch_structure_encoder),
        "adapter": grad_norm(core.highres_skeleton_adapter),
        **{f"stage{i}_interaction": grad_norm(core.decoder_structure_blocks[str(i)]
                                             .skeleton_feature_interaction) for i in (2, 3)},
    }
    assert all(value > 0 for value in report["SH_gradient_abs_sum"].values())
    # The explicit stage target takes precedence over a different fallback target.
    alternate = skeleton.roll(40, dims=-1)
    alt_loss, _ = criterion.highres_structure_skeleton_loss(
        stage_outputs, skeleton, dilate, stage_skeleton_gt=alternate)
    _, target = build_soft_stage_skeleton_target(alternate, (128, 128))
    torch.testing.assert_close(alt_loss, .01 * criterion.soft_stage_skeleton_loss(
        core.last_highres_structure_skeleton, target))
    loss, stats = criterion(outputs[0], surface_gt=surface_target, skeleton_gt=skeleton,
                            skeleton_dilate_gt=dilate, stage_outputs=stage_outputs,
                            boundary_logits=outputs[1], skeleton_logits=outputs[2],
                            connectivity_logits=outputs[3])
    for stage, weight in ((2, .004), (3, .006)):
        torch.testing.assert_close(stats[f"stage{stage}_skeleton_weighted"],
                                   weight * stats[f"stage{stage}_skeleton_raw"])
    torch.testing.assert_close(stats["loss_highres_structure_skeleton"], .01 * sh_stats["highres_structure_skeleton_raw"])
    torch.testing.assert_close(loss.detach(), stats["surface_loss"] + stats["stage_structure_loss"]
                               + stats["loss_highres_structure_skeleton"])
    criterion.stage_skeleton_only_loss_factor = 0
    cd_only, _ = criterion.stage_structure_loss(stage_outputs, skeleton, dilate)
    criterion.stage_skeleton_only_loss_factor = .5
    torch.testing.assert_close(stats["stage_structure_loss"] - cd_only.detach(),
                               stats["stage2_skeleton_weighted"] + stats["stage3_skeleton_weighted"])
    report["skeleton_loss_accounting"] = {key: float(value) for key, value in stats.items()
                                          if key in ("stage2_skeleton_raw", "stage2_skeleton_weighted",
                                                     "stage3_skeleton_raw", "stage3_skeleton_weighted",
                                                     "highres_structure_skeleton_raw",
                                                     "highres_structure_skeleton_weighted", "total_loss")}
    model.zero_grad(set_to_none=True)
    surface_loss = criterion.surface_loss(outputs[0], surface_target)[0]
    surface_loss.backward()
    assert grad_norm(core.surface_h_proj) == 0  # expected with zero-initialized delta
    fusion_optimizer = torch.optim.SGD(core.surface_feature_delta.parameters(), lr=.01)
    fusion_optimizer.step()
    model.zero_grad(set_to_none=True)
    del outputs, stage_outputs, stage_heads, loss, sh_loss, alt_loss, cd_only, surface_loss
    outputs = model(x)
    core.last_stage_features["H2"].retain_grad()
    criterion.surface_loss(outputs[0], surface_target)[0].backward()
    report["surface_gradient_after_delta_update"] = {
        "FH2": float(core.last_stage_features["H2"].grad.abs().sum()),
        "H_projection": grad_norm(core.surface_h_proj),
        "encoder": grad_norm(core.prepatch_structure_encoder),
    }
    assert all(value > 0 for value in report["surface_gradient_after_delta_update"].values())
    model.zero_grad(set_to_none=True)
    del outputs
    model.eval()
    with torch.no_grad():
        expected_output = model(x[:1])[0].cpu()
    checkpoint = dict(args=vars(cfg), skeleton_feature_architecture=architecture_metadata(model, cfg),
                      model_config=model.config.dump(), model_state_dict=model.state_dict())
    require_checkpoint_architecture(checkpoint)
    restore_saved_config(cfg, checkpoint, lambda _: False)
    restored_config = restore_model_config(get_config(cfg), checkpoint["model_config"])
    assert restored_config.dump() == model.config.dump()
    try:
        require_checkpoint_architecture({"args": vars(cfg), "model_state_dict": model.state_dict()})
    except ValueError:
        report["legacy_checkpoint_rejected"] = True
    else:
        raise AssertionError("Old checkpoint was accepted")
    with tempfile.TemporaryDirectory(prefix="skeleton_checkpoint_check_") as tmp:
        path = Path(tmp) / "roundtrip.pth"
        torch.save(checkpoint, path)
        restored = torch.load(path, map_location="cpu", weights_only=False)
        require_checkpoint_architecture(restored)
        # Reload the complete serialized state into the same architecture after changing a weight.
        with torch.no_grad():
            core.surface_feature_delta.weight.add_(1)
        model.load_state_dict(restored["model_state_dict"], strict=True)
        with torch.no_grad():
            actual = model(x[:1])[0].cpu()
        torch.testing.assert_close(actual, expected_output, rtol=0, atol=0)
        assert restored["skeleton_feature_architecture"] == checkpoint["skeleton_feature_architecture"]
    report["checkpoint_roundtrip_exact"] = True
    report["persistent_worker_crop_coords_epochs_0_1_2_0"] = check_crop_workers()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print("[CHECK] all implementation checks passed; no formal training started", flush=True)


if __name__ == "__main__":
    main()
