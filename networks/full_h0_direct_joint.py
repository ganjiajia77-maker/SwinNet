"""Joint H0/repair objective with directly supervised unscaled proposal."""
import torch
from networks.direct_candidate import candidate_segmentation_loss
from networks.full_h0_joint import h0_loss
from networks.reliable_directional_repair import dynamic_losses
from networks.v9_selector_transport_training import live_relation_loss


def enable_joint_training(model):
    model.requires_grad_(True)
    # Scalar gate and scalar relation channel do not participate in source-choice dynamics.
    model.original.gate_relation.delta.requires_grad_(False)
    model.original.relations.direction_head.scalar.requires_grad_(False)
    return model


def joint_loss(model, criterion, result, batch, relation_config, relation_weight, seed, source_losses=None):
    mask, skeleton = batch['mask'], batch['skeleton']
    loss_cfg=dict(source_losses or {})
    segmentation_cfg={key:loss_cfg[key] for key in ('lambda_bg','lambda_missing','missing_threshold') if key in loss_cfg}
    original, _ = h0_loss(criterion, result['h0_outputs'], batch)
    old_surface = criterion.surface_loss(result['z0'], mask)[0]
    h0_aux = original - old_surface
    def segmentation(logits):
        return candidate_segmentation_loss(dict(result, b_logit=logits), mask, skeleton, **segmentation_cfg)[0]
    seg0 = segmentation(result['z0'])
    seg1 = segmentation(result['z1'])
    segb = segmentation(result['b_logit'])
    loss_cfg.update(lambda_cont=.2,return_live_terms=True)
    _, cont_values = dynamic_losses(result, mask, skeleton, **loss_cfg)
    cont = cont_values['cont_live']
    relation, info = live_relation_loss(result, mask, skeleton, relation_config, seed)
    total = .5*seg0 + .5*seg1 + .25*segb + h0_aux + .2*cont + relation_weight*relation
    return total, {'seg_p0':seg0,'seg_p1':seg1,'seg_b':segb,'h0_aux':h0_aux,
                   'cont':cont,'relation':relation,'matching':info['matching']}
