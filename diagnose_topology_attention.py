import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from config import get_config
from datasets.dataset_road_skeleton import RoadSkeletonDataset
from losses.road_losses import SurfaceStructureLoss
from networks.vision_transformer import (
    SwinUnet,
    get_topology_coefficients,
    print_topology_coefficients,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--root_path", default="./data1")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--output_dir", default="./topology_diagnostics")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--source_patch_size", type=int, default=1024)
    parser.add_argument(
        "--cfg",
        default="./configs/swin_tiny_patch4_window7_224_lite.yaml",
    )
    parser.add_argument("--final_topology_eta_init", type=float, default=0.005)

    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache_mode", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--accumulation_steps", type=int, default=0)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--amp_opt_level", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    return parser.parse_args()


def save_probability_map(probability, path):
    array = probability.detach().float().cpu().clamp(0.0, 1.0).numpy()
    Image.fromarray((array * 255.0).round().astype(np.uint8)).save(path)


def aggregate_diagnostics(records):
    if not records:
        return {}

    summary = {}
    for key in records[0]:
        values = [record[key] for record in records]
        if key.endswith("_max"):
            summary[key] = max(values)
        elif key.endswith("_min"):
            summary[key] = min(values)
        else:
            summary[key] = float(np.mean(values))
    return summary


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    config = get_config(args)
    model = SwinUnet(
        config=config,
        img_size=args.img_size,
        num_classes=1,
        use_asterisk=True,
        return_skeleton=True,
        final_topology_eta_init=args.final_topology_eta_init,
    ).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    core = model.swin_unet
    core.guided_head.final_topology_attention.capture_diagnostics = True

    dataset = RoadSkeletonDataset(
        root_dir=args.root_path,
        split=args.split,
        image_size=args.img_size,
        source_patch_size=args.source_patch_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    final_topology_records = []
    saved_samples = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["image"].to(device))
            stage_outputs = outputs[4]

            final_topology_records.append(
                core.guided_head.final_topology_attention.last_diagnostics
            )

            for batch_index, case_name in enumerate(batch["case_name"]):
                if saved_samples >= args.num_samples:
                    break
                for stage_output in stage_outputs:
                    stage = str(stage_output["stage"])
                    skeleton_prob = torch.sigmoid(
                        stage_output["skeleton"][batch_index, 0]
                    )
                    connectivity_prob = torch.sigmoid(
                        stage_output["connectivity"][batch_index]
                    )
                    conn_strength = connectivity_prob.topk(
                        k=2,
                        dim=0,
                    ).values.mean(dim=0)
                    save_probability_map(
                        skeleton_prob,
                        os.path.join(
                            args.output_dir,
                            f"{case_name}_stage{stage}_skeleton_prob_"
                            f"{skeleton_prob.shape[-1]}.png",
                        ),
                    )
                    save_probability_map(
                        conn_strength,
                        os.path.join(
                            args.output_dir,
                            f"{case_name}_stage{stage}_conn_strength_"
                            f"{conn_strength.shape[-1]}.png",
                        ),
                    )
                saved_samples += 1

            if saved_samples >= args.num_samples:
                break

    criterion = SurfaceStructureLoss(
        surface_dice_weight=0.5,
        skeleton_dice_weight=1.0,
        skeleton_weight=0.02,
        connectivity_weight=0.03,
        connectivity_erode_kernel_size=1,
        skeleton_cldice_weight=0.01,
        skeleton_cldice_iterations=10,
        boundary_weight=0.03,
        boundary_radius=1,
        stage_structure_weights=(0.0, 0.0, 0.0, 0.0),
        stage_connectivity_factor=0.5,
        stage_distill_weights=(0.0, 0.0),
        stage_distill_connectivity_factor=0.5,
    )
    guided_head = core.guided_head
    report = {
        "checkpoint": os.path.abspath(args.model_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_topology_version": checkpoint.get(
            "topology_attention_version",
            "legacy-unrecorded",
        ),
        "topology_coefficients": get_topology_coefficients(model),
        "decoder_structure_gate_count": len(core.decoder_structure_blocks),
        "final_topology_diagnostics": aggregate_diagnostics(
            final_topology_records
        ),
        "final_head": {
            "skeleton_head": hasattr(guided_head, "skeleton_head"),
            "connectivity_head": hasattr(guided_head, "connectivity_head"),
            "structure_fusion": hasattr(guided_head, "structure_fusion"),
            "structure_residual": hasattr(guided_head, "structure_residual"),
            "boundary_head": hasattr(guided_head, "boundary_head"),
            "boundary_residual": hasattr(guided_head, "boundary_residual"),
            "surface_head": hasattr(guided_head, "surface_head"),
            "structure_alpha": float(guided_head.alpha.detach().cpu()),
            "boundary_beta": float(guided_head.beta.detach().cpu()),
        },
        "loss_weights": {
            "surface_dice": criterion.surface_loss.dice_weight,
            "final_skeleton": criterion.skeleton_weight,
            "final_connectivity": criterion.connectivity_weight,
            "final_skeleton_cldice": criterion.skeleton_cldice_weight,
            "boundary": criterion.boundary_weight,
            "stage_structure": criterion.stage_structure_weights,
            "stage_distill": criterion.stage_distill_weights,
        },
        "native_stage_maps_saved": saved_samples,
    }

    report_path = os.path.join(args.output_dir, "topology_diagnostics.json")
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=True)

    print_topology_coefficients(model)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"Diagnostic report: {report_path}")


if __name__ == "__main__":
    main()
