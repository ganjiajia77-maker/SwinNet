"""Check same-initialization, no-feedback S2/S3 auxiliary supervision at 512/batch2."""
import argparse
import copy
import json
import runpy
import shlex
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import get_config
from losses.road_losses import build_soft_stage_skeleton_target
from networks.surface_only import skeleton_aux_metadata, require_skeleton_aux_checkpoint
from networks.vision_transformer import SwinUnet, load_topology_checkpoint_state
from evaluation.road_comparison.build_skeleton_aux_mass_manifest import EXPECTED, verify_checkpoint


REFERENCE = dict(return_skeleton=True, structure_profile="stage23_boundary_0626",
                 enable_highres_structure_stream=True, enable_e128_stage_fusion=True,
                 enable_h3_surface_fusion=True, remove_stage2_pre_topology_source=True,
                 enable_global_topology=True, stage_skeleton_mode="direct", surface_only=True)


def runner_configuration():
    script = (ROOT / "scripts/run_e202_skeleton_aux_mass1500.sh").read_text(encoding="utf-8")
    command = script.split("  python -u train_image.py ", 1)[1].split("    2>&1", 1)[0]
    command = command.replace("\\\n", " ").replace('"${resume_args[@]}"', "")
    for name, value in {"$DATA": "unused", "$RESULTS": "unused", "$RUN": "config_check",
                        "$CFG": str(ROOT / "configs/swin_tiny_patch4_window7_224_lite.yaml"),
                        "$PRETRAIN": "not_loaded_in_config_check.pth", "$SEED": "1234"}.items():
        command = command.replace(name, value)
    with mock.patch.object(sys, "argv", ["train_image.py", *shlex.split(command)]):
        trainer = runpy.run_path(str(ROOT / "train_image.py"), run_name="aux_configuration_check")
        trainer["apply_structure_profile_defaults"](trainer["args"])
    args = trainer["args"]
    mismatch = {key: [getattr(args, key, None), value] for key, value in EXPECTED.items()
                if getattr(args, key, None) != value}
    assert not mismatch, mismatch
    return args, trainer


def grad_sum(module):
    return sum(float(p.grad.detach().abs().sum()) for p in module.parameters() if p.grad is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain_ckpt", type=Path)
    parser.add_argument("--output_json", type=Path, default=ROOT / "skeleton_aux_validation.json")
    options = parser.parse_args()
    assert torch.cuda.is_available(), "This 512/batch2 implementation check requires CUDA"
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args, trainer = runner_configuration()
    config = get_config(args)
    if options.pretrain_ckpt:
        config.defrost()
        config.MODEL.PRETRAIN_CKPT = str(options.pretrain_ckpt)
        config.freeze()
    torch.manual_seed(1234)
    baseline = SwinUnet(config, num_classes=1, **REFERENCE)
    baseline_rng = torch.get_rng_state()
    torch.manual_seed(1234)
    aux = SwinUnet(config, num_classes=1, skeleton_aux_only=True, **REFERENCE)
    assert torch.equal(baseline_rng, torch.get_rng_state()), "Head wrapping consumed RNG"
    before, after = baseline.state_dict(), aux.state_dict()
    assert all(torch.equal(value, after[name]) for name, value in before.items())
    assert all(name.startswith("swin_unet.skeleton_aux_heads.") for name in set(after) - set(before))
    loaded_counts = None
    if options.pretrain_ckpt:
        baseline_loaded = baseline.load_from(config)
        aux_loaded = aux.load_from(config)
        assert baseline_loaded == aux_loaded
        assert all(torch.equal(value, aux.state_dict()[name])
                   for name, value in baseline.state_dict().items())
        loaded_counts = len(aux_loaded)
    core = aux.swin_unet
    assert [len(layer.blocks) for layer in core.layers] == [2, 2, 6, 2]
    assert [len(layer.blocks) for layer in core.layers_up[1:]] == [6, 2, 2]
    assert len(core.decoder_structure_blocks) == 0 and len(core.dca_blocks) == 2
    assert core.global_topology is None and core.prepatch_structure_encoder is None
    assert not core.enable_global_topology and not core.enable_highres_structure_stream
    assert not core.enable_e128_stage_fusion and not core.enable_h3_surface_fusion
    assert not core.use_stage3_global_context
    for name, parameter in aux.named_parameters():
        assert not any(token in name for token in (
            "connectivity_head", "direction_head", "gate_branch", "structure_gate", "feature_residual",
            "previous_structure_fusion", "global_topology", "highres_structure", "road_attention",
        )), name
    head2, head3 = core.skeleton_aux_heads["2"], core.skeleton_aux_heads["3"]
    assert head2.structure_branch[0].block[0].weight.data_ptr() != head3.structure_branch[0].block[0].weight.data_ptr()
    assert head2.skeleton_head.out.weight.data_ptr() != head3.skeleton_head.out.weight.data_ptr()
    report = dict(
        surface_baseline="eb2da24d768b7f643e4fb288e3736693862c1fe8",
        shared_initial_state_exact=True, constructor_rng_state_exact=True,
        loaded_imagenet_tensors=loaded_counts,
        baseline_parameters=sum(p.numel() for p in baseline.parameters()),
        auxiliary_parameters=sum(p.numel() for p in aux.parameters()),
        encoder_swin_depths=[2, 2, 6, 2], decoder_swin_depths=[6, 2, 2],
    )
    args_plain = copy.copy(args)
    args_plain.skeleton_aux_only = False
    criterion = trainer["build_criterion"](args, trainer["get_final_loss_weights"](args), "cuda")
    plain_criterion = trainer["build_criterion"](args_plain, trainer["get_final_loss_weights"](args_plain), "cuda")
    assert criterion.stage_structure_weights == (0., 0., .008, .012)
    assert plain_criterion.stage_structure_weights == (0., 0., 0., 0.)
    assert criterion.skeleton_weight == criterion.connectivity_weight == criterion.highres_structure_skeleton_weight == 0
    torch.manual_seed(9876)
    image = torch.randn(2, 3, 512, 512, device="cuda")
    skeleton = torch.zeros(2, 1, 512, 512, device="cuda")
    skeleton[:, :, 250, 25:480] = 1
    skeleton[:, :, 25:480, 245] = 1
    dilate = torch.nn.functional.max_pool2d(skeleton, 3, stride=1, padding=1)
    mask = torch.nn.functional.max_pool2d(skeleton, 9, stride=1, padding=4)
    # Compare both eval and train forwards. Auxiliary heads have no stochastic ops.
    expected = {}
    baseline.cuda()
    for training in (False, True):
        baseline.train(training)
        torch.manual_seed(321)
        with torch.no_grad():
            expected[training] = baseline(image)[0].cpu()
    baseline.cpu()
    aux.cuda()
    for training in (False, True):
        aux.train(training)
        torch.manual_seed(321)
        with torch.no_grad():
            outputs = aux(image)
        torch.testing.assert_close(outputs[0].cpu(), expected[training], rtol=0, atol=0)
        assert outputs[1:4] == (None, None, None)
        assert [item["stage"] for item in outputs[4]] == [2, 3]
        assert all(set(item) == {"stage", "skeleton", "refinement_step", "stage_loss_scale"}
                   for item in outputs[4])
        assert all(tuple(item["skeleton"].shape) == (2, 1, 128, 128) for item in outputs[4])
    report["surface_logits_exact_eval_and_train"] = True
    report["head_output_shapes"] = {"S2": [2, 1, 128, 128], "S3": [2, 1, 128, 128]}
    aux.eval()
    with torch.no_grad():
        before_change = aux(image)[0].clone()
        head2.skeleton_head.out.weight.add_(2)
        head3.skeleton_head.out.bias.sub_(1)
        after_change = aux(image)[0]
        torch.testing.assert_close(before_change, after_change, rtol=0, atol=0)
        head2.skeleton_head.out.weight.sub_(2)
        head3.skeleton_head.out.bias.add_(1)
    report["changing_auxiliary_predictions_cannot_change_surface"] = True
    aux.train()
    outputs = aux(image)
    loss, details = criterion(outputs[0], surface_gt=mask, skeleton_gt=skeleton,
                              skeleton_dilate_gt=dilate, stage_outputs=outputs[4])
    surface_loss = plain_criterion.surface_loss(outputs[0], mask)[0]
    _, target = build_soft_stage_skeleton_target(skeleton, (128, 128))
    raw2 = criterion.soft_stage_skeleton_loss(outputs[4][0]["skeleton"], target)
    raw3 = criterion.soft_stage_skeleton_loss(outputs[4][1]["skeleton"], target)
    weighted = .008 * raw2 + .012 * raw3
    torch.testing.assert_close(loss, surface_loss + weighted, rtol=0, atol=0)
    torch.testing.assert_close(details["stage_structure_loss"], weighted.detach(), rtol=0, atol=0)
    report["exact_loss_decomposition"] = dict(surface=float(surface_loss.detach()),
                                             S2_raw=float(raw2.detach()), S3_raw=float(raw3.detach()),
                                             S2_weighted=float((.008*raw2).detach()),
                                             S3_weighted=float((.012*raw3).detach()), total=float(loss.detach()))
    # The original readout starts at zero; update it once before judging shared gradients.
    optimizer = torch.optim.SGD(core.skeleton_aux_heads.parameters(), lr=.01)
    weighted.backward()
    optimizer.step()
    aux.zero_grad(set_to_none=True)
    del outputs, loss, details, surface_loss, weighted, raw2, raw3
    outputs = aux(image)
    loss2 = criterion.soft_stage_skeleton_loss(outputs[4][0]["skeleton"], target)
    loss3 = criterion.soft_stage_skeleton_loss(outputs[4][1]["skeleton"], target)
    (.008*loss2 + .012*loss3).backward()
    report["S_only_shared_gradient_abs_sum_after_readout_update"] = {
        "encoder_stage1": grad_sum(core.layers[0]), "decoder_stage2": grad_sum(core.layers_up[2]),
        "decoder_stage3": grad_sum(core.layers_up[3]),
    }
    assert all(value > 0 for value in report["S_only_shared_gradient_abs_sum_after_readout_update"].values())
    assert all(p.grad is None for p in core.guided_head.parameters())
    report["S_only_loss_has_no_surface_head_gradient"] = True
    aux.zero_grad(set_to_none=True)
    del outputs, loss2, loss3
    for stage, head in ((2, head2), (3, head3)):
        head.eval()
        feature = torch.randn(1, 96, 8, 8, device="cuda", requires_grad=True)
        head.gradient_ratio = .5
        grad_half = torch.autograd.grad(head(feature).sum(), feature)[0]
        head.gradient_ratio = 1.
        grad_full = torch.autograd.grad(head(feature).sum(), feature)[0]
        head.gradient_ratio = .5
        assert float(grad_full.abs().sum()) > 0
        torch.testing.assert_close(grad_half, .5*grad_full, rtol=0, atol=0)
    report["input_gradient_ratio_exact_half_both_heads"] = True
    aux.eval()
    with torch.no_grad():
        expected_logits = aux(image[:1])[0].cpu()
    checkpoint = dict(epoch=0, args=vars(args), model_state_dict=aux.state_dict(), ema_state_dict=None,
                      skeleton_aux_protocol=skeleton_aux_metadata(core, args))
    with tempfile.TemporaryDirectory(prefix="e202_skeleton_aux_") as directory:
        path = Path(directory) / "roundtrip.pth"
        torch.save(checkpoint, path)
        verify_checkpoint(path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        require_skeleton_aux_checkpoint(loaded)
        with torch.no_grad():
            core.guided_head.surface_head.weight.add_(1)
        load_topology_checkpoint_state(aux, loaded["model_state_dict"], "test", strict=True)
        with torch.no_grad():
            actual_logits = aux(image[:1])[0].cpu()
        torch.testing.assert_close(expected_logits, actual_logits, rtol=0, atol=0)
    try:
        require_skeleton_aux_checkpoint(dict(args=dict(surface_only=True), model_state_dict=baseline.state_dict()))
    except ValueError:
        report["trained_surface_baseline_resume_rejected"] = True
    else:
        raise AssertionError("Surface-only checkpoint was accepted as a complete auxiliary resume")
    try:
        load_topology_checkpoint_state(aux, baseline.state_dict(), "test", strict=True)
    except RuntimeError:
        report["strict_cross_variant_state_load_rejected"] = True
    else:
        raise AssertionError("Missing auxiliary heads were silently loaded")
    report["checkpoint_roundtrip_exact"] = True
    options.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print("[CHECK] auxiliary-only implementation passed; no formal training started", flush=True)


if __name__ == "__main__":
    main()
