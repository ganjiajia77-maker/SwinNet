import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0, pos_weight=None):
        super().__init__()

        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight], dtype=torch.float32)

        self.register_buffer("pos_weight", pos_weight)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
        )
        dice = self.dice(logits, targets)
        loss = self.bce_weight * bce + self.dice_weight * dice

        return loss, bce.detach(), dice.detach()


class SurfaceSkeletonLoss(nn.Module):
    def __init__(
        self,
        surface_dice_weight=0.5,
        skeleton_dice_weight=1.0,
        skeleton_weight=0.3,
        surface_pos_weight=None,
        skeleton_pos_weight=None,
    ):
        super().__init__()

        self.surface_loss = BCEDiceLoss(
            dice_weight=surface_dice_weight,
            bce_weight=1.0,
            pos_weight=surface_pos_weight,
        )
        self.skeleton_loss = BCEDiceLoss(
            dice_weight=skeleton_dice_weight,
            bce_weight=1.0,
            pos_weight=skeleton_pos_weight,
        )
        self.skeleton_weight = skeleton_weight

    def forward(self, surface_logits, skeleton_logits, surface_gt, skeleton_gt):
        loss_surface, bce_surface, dice_surface = self.surface_loss(surface_logits, surface_gt)
        loss_skeleton, bce_skeleton, dice_skeleton = self.skeleton_loss(skeleton_logits, skeleton_gt)
        
        # Total loss: surface + skeleton supervision via skeleton-guided attention
        total_loss = loss_surface + self.skeleton_weight * loss_skeleton

        loss_dict = {
            "total_loss": total_loss.detach(),
            "surface_loss": loss_surface.detach(),
            "skeleton_loss": loss_skeleton.detach(),
            "surface_bce": bce_surface,
            "surface_dice": dice_surface,
            "skeleton_bce": bce_skeleton,
            "skeleton_dice": dice_skeleton,
        }

        return total_loss, loss_dict


def binary_metrics_from_logits(logits, target, threshold=0.5, eps=1e-7):
    prob = torch.sigmoid(logits)
    pred = (prob > threshold).float()
    target = (target > 0.5).float()

    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "iou": iou.item(),
        "f1": f1.item(),
        "precision": precision.item(),
        "recall": recall.item(),
    }
