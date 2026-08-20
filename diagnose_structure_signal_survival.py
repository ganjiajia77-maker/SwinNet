import argparse
import os
import sys
from types import MethodType

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from networks.swin_transformer_unet_skip_expand_decoder_sys import (
    HighResStructureEncoder,
    map_to_token,
    token_to_map,
)
from networks.vision_transformer import (
    STRUCTURE_PROFILE_FULL,
    STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626,
    SwinUnet as ViT_seg,
    load_topology_checkpoint_state,
)


def relative_diff(normal, zero):
    normal = normal.detach().float()
    zero = zero.detach().float()
    return float(torch.linalg.vector_norm(normal - zero) / (torch.linalg.vector_norm(normal) + 1e-8))


def patch_model_for_survival(model, mode):
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    if hasattr(swin, "prepatch_structure_encoder"):
        delattr(swin, "prepatch_structure_encoder")
    swin.highres_structure_encoder = HighResStructureEncoder(
        in_channels=swin.embed_dim,
        struct_channels=swin.highres_structure_channels,
    )
    swin.highres_structure_source = "stage1-survival"
    swin._survival_mode = mode
    swin._survival_trace = {}

    def trace(self, name, value):
        self._survival_trace[name] = value.detach().cpu()

    def build_highres_from_stage1(self, stage1_tokens):
        if not self.enable_highres_structure_stream or stage1_tokens is None:
            return None, None
        stage1_map = token_to_map(stage1_tokens, self.patches_resolution[0], self.patches_resolution[1])
        z_struct = self.highres_structure_encoder(stage1_map)
        skeleton_logits = self.highres_structure_skeleton_head(z_struct)
        return z_struct, skeleton_logits

    def apply_fusion(self, x, z_struct, stage, target_hw):
        if z_struct is None or not self._highres_structure_stage_enabled(stage):
            return x
        feature_map = token_to_map(x, target_hw[0], target_hw[1])
        trace(self, f"stage{stage}_fusion_input", feature_map)
        z_for_surface = z_struct.detach()
        if getattr(self, "_survival_mode", "normal") == "zero_both":
            z_for_surface = torch.zeros_like(z_for_surface)
        feature_map = self.highres_structure_fusion[str(stage)](feature_map, z_for_surface)
        trace(self, f"stage{stage}_fusion_output", feature_map)
        return map_to_token(feature_map)

    def forward_up_features_trace(
        self,
        x,
        x_downsample,
        bottleneck_tokens=None,
        gt_skeleton=None,
        topology_alpha_scale=1.0,
        teacher_forcing_ratio=0.0,
        z_struct=None,
    ):
        structure_outputs = []
        if bottleneck_tokens is None:
            bottleneck_tokens = x
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                pass
            elif inx == 1:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
            else:
                skip = x_downsample[3 - inx]
                h = self.patches_resolution[0] // (2 ** (3 - inx))
                w = self.patches_resolution[1] // (2 ** (3 - inx))
                skip_map = token_to_map(skip, h, w)
                x_map = token_to_map(x, h, w)
                block_idx = inx - 2
                skip_msce = self.msce_blocks[block_idx](skip_map)
                skip_refined = self.dca_blocks[block_idx](deep=x_map, shallow=skip_msce)
                skip_refined = map_to_token(skip_refined)
                x = torch.cat([x, skip_refined], -1)
                x = self.concat_back_dim[inx](x)
                trace(self, f"stage{inx}_after_skip_concat", token_to_map(x, h, w))

            decoder_structure_gate_enabled = (
                self._decoder_structure_enabled(inx)
                and isinstance(layer_up, type(self.layers_up[inx]))
                and inx in (2, 3)
            )
            # Keep the branch condition semantically identical for current model:
            decoder_structure_gate_enabled = (
                self._decoder_structure_enabled(inx)
                and layer_up.__class__.__name__ == "BasicLayer_up"
                and inx in (2, 3)
            )
            topology_enabled = self._stage_topology_enabled(inx)
            if decoder_structure_gate_enabled:
                decoder_skeleton_disabled = self._decoder_skeleton_disabled(inx)
                input_height, input_width = layer_up.input_resolution
                x_map = token_to_map(x, input_height, input_width)
                (
                    _,
                    skeleton_0,
                    connectivity_0,
                    direction_0,
                    structure_gate_0,
                    roadness_0,
                ) = self._run_decoder_structure_block(
                    x_map,
                    inx,
                    bottleneck_tokens,
                    block_stage=1 if inx == 2 else inx,
                    apply_feature_refinement=False,
                    disable_skeleton_prediction=decoder_skeleton_disabled,
                )
                if decoder_skeleton_disabled:
                    skeleton_used = None
                    connectivity_used = self._decoder_connectivity_used(connectivity_0, teacher_forcing_ratio)
                else:
                    skeleton_used, connectivity_used = self._mix_teacher_topology(
                        skeleton_0, connectivity_0, gt_skeleton, teacher_forcing_ratio
                    )
                x = layer_up(
                    x,
                    decoder_skeleton_prob=skeleton_used,
                    decoder_connectivity_prob=connectivity_used,
                    decoder_direction_prob=direction_0,
                )
                output_scale = 2 ** max(2 - inx, 0)
                output_height = self.patches_resolution[0] // output_scale
                output_width = self.patches_resolution[1] // output_scale
                trace(self, f"stage{inx}_after_layer_up", token_to_map(x, output_height, output_width))
                x = self._apply_highres_structure_fusion(x, z_struct, inx, (output_height, output_width))
                self._append_structure_output(
                    structure_outputs,
                    stage=inx,
                    skeleton=skeleton_0,
                    connectivity=connectivity_0,
                    direction=direction_0,
                    structure_gate=structure_gate_0,
                    roadness=roadness_0,
                    refinement_step=0,
                    stage_loss_scale=0.5,
                )
                x_map = token_to_map(x, output_height, output_width)
                (
                    x_map,
                    skeleton_i,
                    connectivity_i,
                    direction_i,
                    structure_gate_i,
                    roadness_i,
                ) = self._run_decoder_structure_block(
                    x_map,
                    inx,
                    bottleneck_tokens,
                    apply_feature_refinement=True,
                    disable_skeleton_prediction=decoder_skeleton_disabled,
                )
                trace(self, f"stage{inx}_after_structure_block", x_map)
                x = map_to_token(x_map)
                self._append_structure_output(
                    structure_outputs,
                    stage=inx,
                    skeleton=skeleton_i,
                    connectivity=connectivity_i,
                    direction=direction_i,
                    structure_gate=structure_gate_i,
                    roadness=roadness_i,
                    refinement_step=1,
                    stage_loss_scale=1.0,
                )
                continue
            elif topology_enabled:
                raise RuntimeError("This survival script expects stage_topology_stages=none for this checkpoint.")
            else:
                decoder_skeleton_disabled = self._decoder_skeleton_disabled(inx)
                x = layer_up(x)
                output_scale = 2 ** max(2 - inx, 0)
                output_height = self.patches_resolution[0] // output_scale
                output_width = self.patches_resolution[1] // output_scale
                x = self._apply_highres_structure_fusion(x, z_struct, inx, (output_height, output_width))
                x_map = token_to_map(x, output_height, output_width)
                (
                    x_map,
                    skeleton_i,
                    connectivity_i,
                    direction_i,
                    structure_gate_i,
                    roadness_i,
                ) = self._run_decoder_structure_block(
                    x_map,
                    inx,
                    bottleneck_tokens,
                    disable_skeleton_prediction=decoder_skeleton_disabled,
                )
                x = map_to_token(x_map)
                self._append_structure_output(
                    structure_outputs,
                    stage=inx,
                    skeleton=skeleton_i,
                    connectivity=connectivity_i,
                    direction=direction_i,
                    structure_gate=structure_gate_i,
                    roadness=roadness_i,
                )

        x = self.norm_up(x)
        trace(self, "final_decoder_feature", token_to_map(x, self.patches_resolution[0], self.patches_resolution[1]))
        return x, structure_outputs

    def forward_trace(self, x, gt_skeleton=None, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0):
        self._survival_trace = {}
        x, x_downsample, road_attentions, stage1_tokens = self.forward_features(x)
        z_struct, highres_structure_skeleton = self._build_highres_structure_outputs(stage1_tokens)
        x, structure_outputs = self.forward_up_features(
            x,
            x_downsample,
            bottleneck_tokens=x,
            gt_skeleton=gt_skeleton,
            topology_alpha_scale=topology_alpha_scale,
            teacher_forcing_ratio=teacher_forcing_ratio,
            z_struct=z_struct,
        )
        if self.return_skeleton and highres_structure_skeleton is not None:
            structure_outputs.append(
                {"stage": "highres_structure", "highres_structure_skeleton": highres_structure_skeleton}
            )
        if self.return_skeleton and road_attentions:
            structure_outputs.extend(road_attentions)
        x = self.up_x4(x, structure_outputs=structure_outputs if self.return_skeleton else None)
        surface_logits = x[0] if isinstance(x, tuple) else x
        trace(self, "surface_logits", surface_logits)
        if self.return_skeleton and isinstance(x, tuple):
            x = (*x, structure_outputs)
        return x

    swin._build_highres_structure_outputs = MethodType(build_highres_from_stage1, swin)
    swin._apply_highres_structure_fusion = MethodType(apply_fusion, swin)
    swin.forward_up_features = MethodType(forward_up_features_trace, swin)
    swin.forward = MethodType(forward_trace, swin)


def build_model(args, checkpoint, mode):
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
    )
    patch_model_for_survival(model, mode)
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=False,
    )
    return model.to(args.device).eval()


def run_mode(args, checkpoint, images, mode):
    model = build_model(args, checkpoint, mode)
    with torch.no_grad():
        model(images)
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    trace = dict(swin._survival_trace)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
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
    parser.add_argument(
        "--structure_profile",
        type=str,
        default=STRUCTURE_PROFILE_FULL,
        choices=[STRUCTURE_PROFILE_FULL, STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626],
    )
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23", choices=["stage2", "stage3", "stage23"])
    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location="cpu")
    if isinstance(checkpoint.get("args"), dict):
        saved_args = checkpoint["args"]
        args.structure_profile = saved_args.get("structure_profile", args.structure_profile)
        args.disable_msfe_skip = bool(saved_args.get("disable_msfe_skip", args.disable_msfe_skip))
        args.enable_highres_structure_stream = bool(saved_args.get("enable_highres_structure_stream", args.enable_highres_structure_stream))
        args.highres_structure_channels = int(saved_args.get("highres_structure_channels", args.highres_structure_channels))
        args.highres_structure_fuse_stages = str(saved_args.get("highres_structure_fuse_stages", args.highres_structure_fuse_stages))

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    batch = next(iter(loader))
    images = batch["image"].to(args.device)

    print(f"Using device: {args.device}")
    print(f"Checkpoint: {args.model_path}")
    print(f"Batch size: {images.shape[0]}, image={tuple(images.shape)}")

    normal = run_mode(args, checkpoint, images, "normal")
    zero = run_mode(args, checkpoint, images, "zero_both")

    labels = [
        ("stage2_after_skip_concat", "Stage2 fusion pre-input"),
        ("stage2_after_layer_up", "Stage2 after upsample/block"),
        ("stage2_fusion_output", "Stage2 fusion output"),
        ("stage2_after_structure_block", "After Stage2 structure block"),
        ("stage3_after_skip_concat", "Stage3 fusion pre-input"),
        ("stage3_after_layer_up", "Stage3 after upsample/block"),
        ("stage3_fusion_output", "Stage3 fusion output"),
        ("stage3_after_structure_block", "After Stage3 structure block"),
        ("final_decoder_feature", "Final decoder feature"),
        ("surface_logits", "Surface logits"),
    ]

    print("\n========== Structure Signal Survival ==========")
    for key, label in labels:
        if key not in normal or key not in zero:
            continue
        print(f"{label:<34} {relative_diff(normal[key], zero[key]):.8f}")


if __name__ == "__main__":
    main()
