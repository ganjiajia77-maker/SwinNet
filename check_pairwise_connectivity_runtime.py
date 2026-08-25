import argparse

from config import get_config
from networks.vision_transformer import SwinUnet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--structure_profile", type=str, default="stage23_boundary_0626")
    parser.add_argument("--disable_msfe_skip", action="store_true")
    parser.add_argument("--enable_highres_structure_stream", action="store_true")
    parser.add_argument("--highres_structure_channels", type=int, default=64)
    parser.add_argument("--highres_structure_fuse_stages", type=str, default="stage23")
    parser.add_argument("--highres_structure_fusion_mode", type=str, default="stage23")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
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
    parser.add_argument("--n_class", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config(args)
    model = SwinUnet(
        config=config,
        img_size=args.img_size,
        num_classes=args.num_classes,
        return_skeleton=True,
        structure_profile=args.structure_profile,
        use_msfe_skip=not args.disable_msfe_skip,
        enable_highres_structure_stream=args.enable_highres_structure_stream,
        highres_structure_channels=args.highres_structure_channels,
        highres_structure_fuse_stages=args.highres_structure_fuse_stages,
        highres_structure_fusion_mode=args.highres_structure_fusion_mode,
    )
    state = model.state_dict()
    has_pairwise = any(".connectivity_head.edge_mlp." in key for key in state)
    has_axis_basis = any(".connectivity_head.axis_basis" in key for key in state)
    has_old_conv = any(key.endswith(".connectivity_head.weight") for key in state)
    print(f"runtime pairwise edge_mlp: {has_pairwise}")
    print(f"runtime axis_basis: {has_axis_basis}")
    print(f"runtime old conv: {has_old_conv}")
    for key, value in state.items():
        if "decoder_structure_blocks.2.connectivity_head" in key:
            print(key, tuple(value.shape))
            break
    if not has_pairwise or not has_axis_basis or has_old_conv:
        raise SystemExit("Runtime connectivity head is not the expected pairwise head.")


if __name__ == "__main__":
    main()
