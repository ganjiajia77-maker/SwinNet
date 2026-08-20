import argparse
import os
import sys
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def patch_stage1_highres_sensitivity(model):
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet
    if hasattr(swin, "prepatch_structure_encoder"):
        delattr(swin, "prepatch_structure_encoder")
    swin.highres_structure_encoder = HighResStructureEncoder(
        in_channels=swin.embed_dim,
        struct_channels=swin.highres_structure_channels,
    )
    swin.highres_structure_source = "stage1-sensitivity"
    swin._sensitivity_z_leaves = {}

    def build_highres_from_stage1(self, stage1_tokens):
        if not self.enable_highres_structure_stream or stage1_tokens is None:
            return None, None
        stage1_map = token_to_map(
            stage1_tokens,
            self.patches_resolution[0],
            self.patches_resolution[1],
        )
        z_struct = self.highres_structure_encoder(stage1_map)
        skeleton_logits = self.highres_structure_skeleton_head(z_struct)
        return z_struct, skeleton_logits

    def apply_fusion_with_leaf_z(self, x, z_struct, stage, target_hw):
        if z_struct is None or not self._highres_structure_stage_enabled(stage):
            return x
        feature_map = token_to_map(x, target_hw[0], target_hw[1])
        stage_name = f"stage{int(stage)}"
        z_leaf = z_struct.detach().clone().requires_grad_(True)
        z_leaf.retain_grad()
        self._sensitivity_z_leaves[stage_name] = z_leaf
        feature_map = self.highres_structure_fusion[str(stage)](feature_map, z_leaf)
        return map_to_token(feature_map)

    def forward_stage1_highres(self, x, gt_skeleton=None, topology_alpha_scale=1.0, teacher_forcing_ratio=0.0):
        self._sensitivity_z_leaves = {}
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
                {
                    "stage": "highres_structure",
                    "highres_structure_skeleton": highres_structure_skeleton,
                }
            )
        if self.return_skeleton and road_attentions:
            structure_outputs.extend(road_attentions)
        x = self.up_x4(
            x,
            structure_outputs=structure_outputs if self.return_skeleton else None,
        )
        if self.return_skeleton and isinstance(x, tuple):
            x = (*x, structure_outputs)
        return x

    swin._build_highres_structure_outputs = MethodType(build_highres_from_stage1, swin)
    swin._apply_highres_structure_fusion = MethodType(apply_fusion_with_leaf_z, swin)
    swin.forward = MethodType(forward_stage1_highres, swin)


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
    )
    patch_stage1_highres_sensitivity(model)
    load_topology_checkpoint_state(
        model,
        checkpoint["model_state_dict"],
        checkpoint.get("topology_attention_version", "legacy-unrecorded"),
        strict=False,
    )
    return model.to(args.device).eval()


def l2_norm(tensor):
    return float(torch.linalg.vector_norm(tensor.detach()).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data1")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--crop_list", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument("--max_batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
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

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model_path, map_location="cpu")
    if isinstance(checkpoint.get("args"), dict):
        saved_args = checkpoint["args"]
        args.structure_profile = saved_args.get("structure_profile", args.structure_profile)
        args.disable_msfe_skip = bool(saved_args.get("disable_msfe_skip", args.disable_msfe_skip))
        args.enable_highres_structure_stream = bool(
            saved_args.get("enable_highres_structure_stream", args.enable_highres_structure_stream)
        )
        args.highres_structure_channels = int(saved_args.get("highres_structure_channels", args.highres_structure_channels))
        args.highres_structure_fuse_stages = str(
            saved_args.get("highres_structure_fuse_stages", args.highres_structure_fuse_stages)
        )

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
        crop_list_path=args.crop_list,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = build_model(args, checkpoint)
    swin = model.module.swin_unet if hasattr(model, "module") else model.swin_unet

    stats = {
        "stage2": {"grad_norm": [], "z_norm": [], "sensitivity": [], "grad_abs_mean": [], "z_abs_mean": []},
        "stage3": {"grad_norm": [], "z_norm": [], "sensitivity": [], "grad_abs_mean": [], "z_abs_mean": []},
    }

    print(f"Using device: {args.device}")
    print(f"Checkpoint: {args.model_path}")
    print(f"Surface sensitivity: split={args.split}, max_batches={args.max_batches}, loss=BCEWithLogits")

    for batch_index, batch in enumerate(tqdm(loader, desc="Sensitivity")):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        model.zero_grad(set_to_none=True)
        images = batch["image"].to(args.device)
        masks = batch["mask"].to(args.device)
        outputs = model(images)
        surface_logits = outputs[0] if isinstance(outputs, tuple) else outputs
        if masks.shape[-2:] != surface_logits.shape[-2:]:
            masks = F.interpolate(masks.float(), size=surface_logits.shape[-2:], mode="nearest")
        loss = F.binary_cross_entropy_with_logits(surface_logits, masks.float())
        leaves = dict(getattr(swin, "_sensitivity_z_leaves", {}))
        grads = torch.autograd.grad(
            loss,
            [leaves[key] for key in ("stage2", "stage3") if key in leaves],
            retain_graph=False,
            allow_unused=True,
        )
        for stage, grad in zip([key for key in ("stage2", "stage3") if key in leaves], grads):
            if grad is None:
                continue
            z = leaves[stage]
            grad_norm = l2_norm(grad)
            z_norm = l2_norm(z)
            stats[stage]["grad_norm"].append(grad_norm)
            stats[stage]["z_norm"].append(z_norm)
            stats[stage]["sensitivity"].append(grad_norm / (z_norm + 1e-8))
            stats[stage]["grad_abs_mean"].append(float(grad.detach().abs().mean().item()))
            stats[stage]["z_abs_mean"].append(float(z.detach().abs().mean().item()))

    print("\nSURFACE -> STRUCTURE SENSITIVITY")
    print("stage    ||dL/dZ||      ||Z||          S=grad/Z      mean|dL/dZ|    mean|Z|")
    for stage in ("stage2", "stage3"):
        row = stats[stage]
        if not row["grad_norm"]:
            print(f"{stage:<8} no samples")
            continue
        print(
            f"{stage:<8} "
            f"{np.mean(row['grad_norm']):<13.8f} "
            f"{np.mean(row['z_norm']):<13.6f} "
            f"{np.mean(row['sensitivity']):<13.10f} "
            f"{np.mean(row['grad_abs_mean']):<13.10f} "
            f"{np.mean(row['z_abs_mean']):<13.8f}"
        )


if __name__ == "__main__":
    main()
