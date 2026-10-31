from .losses.road_losses import SurfaceStructureLoss
from .networks.vision_transformer import STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626, STRUCTURE_PROFILE_STAGE23_BOUNDARY_FINAL_SKE

def get_final_loss_weights(args):
    if args.structure_profile == STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626:
        weights = {'skeleton_weight': 0.0, 'connectivity_weight': 0.0, 'boundary_weight': 0.0}
    elif args.structure_profile == STRUCTURE_PROFILE_STAGE23_BOUNDARY_FINAL_SKE:
        weights = {'skeleton_weight': 0.0, 'connectivity_weight': 0.0, 'boundary_weight': 0.0}
    else:
        weights = {'skeleton_weight': 0.02, 'connectivity_weight': 0.03, 'boundary_weight': 0.0}
    if args.final_skeleton_weight is not None:
        weights['skeleton_weight'] = float(args.final_skeleton_weight)
    if args.final_connectivity_weight is not None:
        if args.structure_profile in {STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626, STRUCTURE_PROFILE_STAGE23_BOUNDARY_FINAL_SKE}:
            print('[WARN] --final_connectivity_weight is ignored for stage23 profiles; final connectivity is removed.', flush=True)
        else:
            weights['connectivity_weight'] = float(args.final_connectivity_weight)
    if args.boundary_weight is not None and float(args.boundary_weight) != 0.0:
        print('[WARN] --boundary_weight is ignored because boundary auxiliary loss is disabled.')
    return weights

def build_criterion(args, loss_weights, device):
    stage2_weight = args.stage2_skeleton_weight
    stage3_weight = args.stage3_skeleton_weight
    boundary_weight = 0.0
    return SurfaceStructureLoss(surface_dice_weight=0.5, skeleton_dice_weight=1.0, skeleton_weight=loss_weights['skeleton_weight'], connectivity_weight=loss_weights['connectivity_weight'], connectivity_erode_kernel_size=1, boundary_weight=boundary_weight, boundary_radius=1, stage_structure_weights=(0.0, 0.0, stage2_weight, stage3_weight), road_attention_weight=args.road_attention_weight, stage_connectivity_factor=args.stage_connectivity_factor, stage_direction_factor=args.stage_direction_factor, highres_structure_skeleton_weight=args.highres_structure_skeleton_weight, use_legacy_stage_connectivity_loss=args.structure_profile in {STRUCTURE_PROFILE_STAGE23_BOUNDARY_0626, STRUCTURE_PROFILE_STAGE23_BOUNDARY_FINAL_SKE}, use_masked_connectivity_center_experiment=args.masked_connectivity_center_experiment, connectivity_pos_weight=args.connectivity_pos_weight, skeleton_pos_weight=args.skeleton_pos_weight, directional_pos_weight_cardinal=args.directional_pos_weight_cardinal, directional_pos_weight_diagonal=args.directional_pos_weight_diagonal, connectivity_focal_gamma=args.connectivity_focal_gamma, surface_focal_gamma=args.surface_focal_gamma, edge_contrastive_margin=args.edge_contrastive_margin, stage_connectivity_factors=(0.0, 0.0, args.stage2_connectivity_factor if args.stage2_connectivity_factor is not None else args.stage_connectivity_factor, args.stage3_connectivity_factor if args.stage3_connectivity_factor is not None else args.stage_connectivity_factor), stage_direction_factors=(0.0, 0.0, args.stage2_direction_factor if args.stage2_direction_factor is not None else args.stage_direction_factor, args.stage3_direction_factor if args.stage3_direction_factor is not None else args.stage_direction_factor), stage_skeleton_loss_factor=args.stage_skeleton_loss_factor, stage_skeleton_only_loss_factor=args.stage_skeleton_only_loss_factor).to(device)
