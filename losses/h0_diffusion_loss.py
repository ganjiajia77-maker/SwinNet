"""Only final surface, original S2/S3 soft-skeleton losses and original D3 loss."""
import torch
from torch.nn import functional as F
from .road_losses import SurfaceStructureLoss, build_soft_stage_skeleton_target


class H0DiffusionLoss(SurfaceStructureLoss):
    def __init__(self):
        super().__init__(surface_focal_gamma=1., surface_dice_weight=.5,
                         skeleton_weight=0., connectivity_weight=0., skeleton_stage_weight=0.,
                         stage_structure_weights=(0., 0., .008, .012),
                         stage_direction_factor=.2, stage_skeleton_loss_factor=1.,
                         stage_skeleton_only_loss_factor=1., skeleton_pos_weight=2.)

    def forward(self, surface_logits, surface_gt, skeleton_gt=None, stage_outputs=None, **unused):
        surface, bce, dice = self.surface_loss(surface_logits, surface_gt)
        total = surface
        terms = {'surface_loss': surface.detach(), 'surface_bce': bce, 'surface_dice': dice,
                 'S2': None, 'S3': None, 'D3': None, 'C2': None, 'D2': None, 'C3': None,
                 'highres_skeleton': None, 'boundary_aux': None, 'global_topology': None}
        if stage_outputs is not None:
            stages = [item for item in stage_outputs if item.get('stage') in (2, 3)]
            if sorted(item['stage'] for item in stages) != [2, 3]:
                raise ValueError('Require exactly one S2 and one S3/D3 output')
            for item in stages:
                index = item['stage']
                if item.get('connectivity') is not None or (index == 2 and item.get('direction') is not None):
                    raise ValueError('C2/D2/C3 must be None, not zero logits')
                if item.get('structure_gate') is not None or item.get('stage_loss_scale', 1.) != 1.:
                    raise ValueError('Old gate is disabled; each stage loss is included once')
                prediction = item['skeleton']
                hard, soft = build_soft_stage_skeleton_target(skeleton_gt, prediction.shape[-2:])
                skeleton_loss = self.soft_stage_skeleton_loss(prediction, soft)
                weight = self.stage_structure_weights[index]
                total = total + weight*skeleton_loss
                terms[f'S{index}'] = skeleton_loss.detach()
                if index == 3:
                    direction = item.get('direction')
                    if direction is None:
                        raise ValueError('D3 prediction is required')
                    target, valid = self.build_direction_target(hard)
                    valid = valid.to(direction) * self._spatial_boundary_mask(direction)
                    cosine = (F.normalize(direction, dim=1, eps=1e-6)*target.to(direction)).sum(1, keepdim=True)
                    direction_loss = ((1.-cosine)*valid).sum()/valid.sum().clamp_min(1.)
                    total = total + weight*self.stage_direction_factor*direction_loss
                    terms['D3'] = direction_loss.detach()
        terms['total_loss'] = total.detach()
        return total, terms
