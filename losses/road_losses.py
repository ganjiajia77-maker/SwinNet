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

        # 原始 BCE
        bce_unreduced = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction='none'
        )

        # 直接取平均 BCE
        bce = bce_unreduced.mean()

        dice = self.dice(logits, targets)
        loss = self.bce_weight * bce + self.dice_weight * dice

        return loss, bce.detach(), dice.detach()


class CenterlineResponseLoss(nn.Module):
    """
    Centerline coverage loss.

    Dilate the ground-truth centerline into a tolerance band first, then
    require the road surface probability to cover that band.
    """
    def __init__(self, dilation_size=5, eps=1e-6):
        super().__init__()
        self.dilation_size = dilation_size
        self.eps = eps

    def dilate_skeleton(self, skeleton):
        """
        skeleton: [B, 1, H, W], value 0/1
        return:   [B, 1, H, W], dilated centerline band
        """
        if self.dilation_size <= 1:
            return skeleton

        padding = self.dilation_size // 2
        band = F.max_pool2d(
            skeleton,
            kernel_size=self.dilation_size,
            stride=1,
            padding=padding,
        )
        return band

    def forward(self, surface_logits, skeleton_gt):
        """
        surface_logits: [B, 1, H, W] or [B, H, W]
        skeleton_gt:    [B, 1, H, W] or [B, H, W], binary skeleton label
        """
        if surface_logits.dim() == 3:
            surface_logits = surface_logits.unsqueeze(1)
        if skeleton_gt.dim() == 3:
            skeleton_gt = skeleton_gt.unsqueeze(1)

        skeleton_gt = skeleton_gt.float()
        surface_prob = torch.sigmoid(surface_logits)

        center_band = self.dilate_skeleton(skeleton_gt)

        denom = center_band.sum(dim=(1, 2, 3)) + self.eps
        log_prob = torch.log(surface_prob.clamp_min(self.eps))
        loss = -(log_prob * center_band).sum(dim=(1, 2, 3)) / denom
        return loss.mean()


class EdgeAwareLoss(nn.Module):
    def __init__(self, edge_width=3, eps=1e-6):
        super().__init__()
        self.edge_width = edge_width
        self.eps = eps

    def extract_edge(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()
        padding = self.edge_width // 2
        dilated = F.max_pool2d(x, kernel_size=self.edge_width, stride=1, padding=padding)
        eroded = 1.0 - F.max_pool2d(1.0 - x, kernel_size=self.edge_width, stride=1, padding=padding)
        return (dilated - eroded).clamp(0.0, 1.0)

    def forward(self, surface_logits, surface_gt):
        if surface_logits.dim() == 3:
            surface_logits = surface_logits.unsqueeze(1)
        if surface_gt.dim() == 3:
            surface_gt = surface_gt.unsqueeze(1)

        surface_prob = torch.sigmoid(surface_logits)
        pred_edge = self.extract_edge(surface_prob)
        gt_edge = self.extract_edge((surface_gt > 0.5).float())

        bce = F.binary_cross_entropy(pred_edge.clamp(self.eps, 1.0 - self.eps), gt_edge)
        intersection = (pred_edge * gt_edge).sum(dim=(1, 2, 3))
        union = pred_edge.sum(dim=(1, 2, 3)) + gt_edge.sum(dim=(1, 2, 3))
        dice = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)
        return bce + dice.mean()


def soft_erode(img):
    erode_h = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    erode_w = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(erode_h, erode_w)


def soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img, iterations=10):
    img = img.clamp(0.0, 1.0)
    skeleton = F.relu(img - soft_open(img))

    for _ in range(iterations):
        img = soft_erode(img)
        delta = F.relu(img - soft_open(img))
        skeleton = skeleton + F.relu(delta - skeleton * delta)

    return skeleton.clamp(0.0, 1.0)


class SoftClDiceLoss(nn.Module):
    def __init__(self, iterations=10, smooth=1.0):
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth

    def forward(self, probs, targets):
        if probs.dim() == 3:
            probs = probs.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        probs = probs.float().clamp(0.0, 1.0)
        targets = (targets > 0.5).float()

        pred_skel = soft_skeletonize(probs, iterations=self.iterations)
        target_skel = soft_skeletonize(targets, iterations=self.iterations)

        dims = (1, 2, 3)
        tprec = (pred_skel * targets).sum(dims) / (pred_skel.sum(dims) + self.smooth)
        tsens = (target_skel * probs).sum(dims) / (target_skel.sum(dims) + self.smooth)
        cldice = (2.0 * tprec * tsens + self.smooth) / (tprec + tsens + self.smooth)
        return (1.0 - cldice).mean()


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


def erode_binary_mask(mask, kernel_size=3):
    if kernel_size <= 1:
        return mask

    padding = kernel_size // 2
    eroded = 1.0 - F.max_pool2d(
        1.0 - mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )
    return eroded.clamp(0.0, 1.0)


def dilate_binary_mask(mask, radius=3):
    if radius <= 0:
        return mask

    kernel_size = 2 * radius + 1
    return F.max_pool2d(
        mask.float(),
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    ).clamp(0.0, 1.0)


def build_connectivity_target(surface_gt, erode_kernel_size=1):
    if surface_gt.dim() == 3:
        surface_gt = surface_gt.unsqueeze(1)

    road = (surface_gt > 0.5).float()
    road = erode_binary_mask(road, kernel_size=erode_kernel_size)

    padded = F.pad(road, (1, 1, 1, 1))
    height, width = road.shape[-2:]
    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    targets = []
    for dy, dx in directions:
        y0 = 1 + dy
        x0 = 1 + dx
        neighbor = padded[:, :, y0:y0 + height, x0:x0 + width]
        targets.append(road * neighbor)

    return torch.cat(targets, dim=1)


class SurfaceStructureLoss(nn.Module):
    def __init__(
        self,
        surface_dice_weight=0.5,
        skeleton_dice_weight=1.0,
        skeleton_weight=0.05,
        connectivity_weight=0.05,
        connectivity_erode_kernel_size=1,
        skeleton_cldice_weight=0.0,
        skeleton_cldice_iterations=10,
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
        self.connectivity_weight = connectivity_weight
        self.connectivity_erode_kernel_size = connectivity_erode_kernel_size
        self.skeleton_cldice_weight = skeleton_cldice_weight
        self.skeleton_cldice_loss = SoftClDiceLoss(
            iterations=skeleton_cldice_iterations,
        )

    def forward(
        self,
        surface_logits,
        skeleton_logits,
        connectivity_logits,
        surface_gt,
        skeleton_gt,
        skeleton_dilate_gt=None,
    ):
        loss_surface, bce_surface, dice_surface = self.surface_loss(surface_logits, surface_gt)
        if skeleton_dilate_gt is None:
            skeleton_dilate_gt = skeleton_gt

        skeleton_pos_weight = (
            self.skeleton_loss.pos_weight.to(skeleton_logits.device)
            if self.skeleton_loss.pos_weight is not None
            else None
        )
        bce_skeleton = F.binary_cross_entropy_with_logits(
            skeleton_logits,
            skeleton_dilate_gt,
            pos_weight=skeleton_pos_weight,
        )
        dice_skeleton = self.skeleton_loss.dice(skeleton_logits, skeleton_gt)
        loss_skeleton = bce_skeleton + self.skeleton_loss.dice_weight * dice_skeleton
        connectivity_gt = build_connectivity_target(
            skeleton_gt,
            erode_kernel_size=self.connectivity_erode_kernel_size,
        ).to(
            device=connectivity_logits.device,
            dtype=connectivity_logits.dtype,
        )
        loss_connectivity = F.binary_cross_entropy_with_logits(
            connectivity_logits,
            connectivity_gt,
        )
        loss_skeleton_cldice = self.skeleton_cldice_loss(
            torch.sigmoid(skeleton_logits),
            skeleton_gt,
        )

        total_loss = (
            loss_surface
            + self.skeleton_weight * loss_skeleton
            + self.connectivity_weight * loss_connectivity
            + self.skeleton_cldice_weight * loss_skeleton_cldice
        )

        loss_dict = {
            "total_loss": total_loss.detach(),
            "surface_loss": loss_surface.detach(),
            "skeleton_loss": loss_skeleton.detach(),
            "connectivity_loss": loss_connectivity.detach(),
            "skeleton_cldice_loss": loss_skeleton_cldice.detach(),
            "surface_bce": bce_surface,
            "surface_dice": dice_surface,
            "skeleton_bce": bce_skeleton,
            "skeleton_dice": dice_skeleton,
        }

        return total_loss, loss_dict


class SurfaceCoverageLoss(nn.Module):
    def __init__(
        self,
        surface_dice_weight=0.5,
        centerline_weight=0.03,
        centerline_dilation_size=5,
        edge_weight=0.05,
        edge_width=3,
        surface_pos_weight=None,
    ):
        super().__init__()

        self.surface_loss = BCEDiceLoss(
            dice_weight=surface_dice_weight,
            bce_weight=1.0,
            pos_weight=surface_pos_weight,
        )
        self.centerline_loss = CenterlineResponseLoss(
            dilation_size=centerline_dilation_size,
        )
        self.edge_loss = EdgeAwareLoss(edge_width=edge_width)

        self.centerline_weight = centerline_weight
        self.edge_weight = edge_weight

    def forward(self, surface_logits, surface_gt, skeleton_gt):
        loss_surface, bce_surface, dice_surface = self.surface_loss(surface_logits, surface_gt)
        loss_centerline = self.centerline_loss(surface_logits, skeleton_gt)
        loss_edge = self.edge_loss(surface_logits, surface_gt)

        total_loss = (
            loss_surface
            + self.centerline_weight * loss_centerline
            + self.edge_weight * loss_edge
        )

        loss_dict = {
            "total_loss": total_loss.detach(),
            "surface_loss": loss_surface.detach(),
            "centerline_loss": loss_centerline.detach(),
            "edge_loss": loss_edge.detach(),
            "surface_bce": bce_surface,
            "surface_dice": dice_surface,
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
