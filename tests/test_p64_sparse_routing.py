import torch

from losses.road_losses import SurfaceStructureLoss
from networks.swin_transformer_unet_skip_expand_decoder_sys import (
    SwinTransformerBlock,
    SwinTransformerSys,
)


def test_shifted_window_scores_original_probability_coordinates():
    block = SwinTransformerBlock(
        dim=16,
        input_resolution=(32, 32),
        num_heads=4,
        window_size=8,
        shift_size=4,
        drop_path=0.0,
    ).eval()
    feature = torch.randn(1, 32 * 32, 16)
    probability = torch.zeros(1, 1, 32, 32)
    probability[0, 0, 7, 7] = 1.0
    with torch.no_grad():
        output, stats = block.forward_sparse_windows(
            feature, probability, threshold=0.5
        )
    assert stats["active_windows"] == 1
    assert stats["total_windows"] == 16
    assert output.shape == feature.shape
    assert torch.equal(output[~stats["active_token_mask"].expand_as(feature)],
                       feature[~stats["active_token_mask"].expand_as(feature)])


def test_p64_bce_dice_contributes_gradient():
    criterion = SurfaceStructureLoss(coarse_road_weight=0.2)
    logits = torch.nn.Parameter(torch.zeros(2, 1, 64, 64))
    target = torch.zeros_like(logits)
    target[:, :, 20:22, 20:60] = 1
    loss, raw = criterion.coarse_road_loss(
        [{"coarse_road_logits": logits}], target, logits
    )
    assert loss.item() > 0
    assert raw.item() > 0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0


def test_dense_sparse_forward_retains_h2_h3_and_surface():
    model = SwinTransformerSys(
        img_size=256,
        embed_dim=24,
        depths=(2, 2, 1, 1),
        num_heads=(3, 3, 3, 3),
        window_size=8,
        drop_path_rate=0.0,
        num_classes=1,
        return_skeleton=True,
        structure_profile="stage23_boundary_0626",
        enable_coarse_road_mask=True,
        enable_psi_directional_descriptor=True,
        sparse_window_compute=True,
        coarse_routing_mode="p64",
        stage_skeleton_mode="direct",
        routing_warmup_epochs=10,
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        model.set_route_epoch(0)
        dense = model(image)
        assert len(model.layers_up[2].blocks) == 2
        assert model.layers_up[2].blocks[1].shift_size == 4
        assert dense[0].shape == (1, 1, 256, 256)
        assert model.last_stage_features["H2"].shape[-2:] == (64, 64)
        assert model.last_stage_features["H3"].shape[-2:] == (64, 64)
        model.set_route_epoch(10)
        model.set_routing_calibration_done(True)
        model.sparse_selection_probability_override = torch.zeros(
            1, 1, 64, 64, device=device
        )
        sparse = model(image)
        assert sparse[0].shape == dense[0].shape
        assert model.last_route_stats["stage2"]["active_windows"] == 0
        assert model.last_route_stats["stage3"]["active_windows"] == 0
        assert model.last_stage_features["H2"].shape[-2:] == (64, 64)
        assert model.last_stage_features["H3"].shape[-2:] == (64, 64)
        model.sparse_selection_probability_override.fill_(1)
        dense_route = model(image)
        assert dense_route[0].shape == dense[0].shape
        assert model.last_route_stats["stage2"]["active_ratio"] == 1.0
        assert model.last_route_stats["stage3"]["active_ratio"] == 1.0
        model.sparse_selection_probability_override.zero_()
        model.sparse_selection_probability_override[:, :, 8:16, 8:16] = 1
        sparse_partial = model(image)
        assert sparse_partial[0].shape == dense[0].shape
        assert 0 < model.last_route_stats["stage2"]["active_windows"]
        assert model.last_route_stats["stage2"]["active_ratio"] < 1
        assert 0 < model.last_route_stats["stage3"]["active_windows"]
        assert model.last_route_stats["stage3"]["active_ratio"] < 1
        for stage in (2, 3):
            assert model.last_route_stats[f"stage{stage}"]["total_windows"] > 0
    model.sparse_selection_probability_override = None
    model.set_route_epoch(0)
    output = model(image)
    p64_target = torch.zeros(1, 1, 64, 64, device=device)
    p64_target[:, :, 12:17, 10:55] = 1
    mask_loss, _ = SurfaceStructureLoss(coarse_road_weight=0.2).coarse_road_loss(
        output[4], p64_target, output[0]
    )
    mask_loss.backward()
    gradient = model.coarse_road_mask_head.out.weight.grad
    assert gradient is not None and gradient.abs().sum().item() > 0
    mask = torch.zeros_like(output[0])
    mask[:, :, 48:72, 40:180] = 1
    criterion = SurfaceStructureLoss(
        coarse_road_weight=0.2,
        stage_structure_weights=(0.0, 0.0, 0.008, 0.012),
    )
    total, detail = criterion(
        surface_logits=output[0],
        skeleton_logits=output[1],
        connectivity_logits=output[2],
        surface_gt=mask,
        skeleton_gt=mask,
        skeleton_dilate_gt=mask,
        stage_outputs=output[4],
        coarse_road_gt=p64_target,
    )
    assert torch.isfinite(total)
    assert detail["loss_coarse_road"].item() > 0
