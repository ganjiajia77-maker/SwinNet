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


def build_boundary_target(surface_gt, radius=1):
    if surface_gt.dim() == 3:
        surface_gt = surface_gt.unsqueeze(1)

    road = (surface_gt > 0.5).float()
    dilated = dilate_binary_mask(road, radius=radius)
    eroded = erode_binary_mask(road, kernel_size=2 * radius + 1)
    return (dilated - eroded).clamp(0.0, 1.0)


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
        boundary_weight=0.01,
        boundary_radius=1,
        skeleton_stage_weight=0.0,
        skeleton_stage_weights=(0.1, 0.2, 0.3, 0.3),
        stage_structure_weights=None,
        stage_connectivity_factor=0.5,
        stage_direction_factor=0.2,
        stage_skeleton_connectivity_s2c_weight=1.0,
        stage_skeleton_connectivity_c2s_weight=0.2,
        road_attention_weight=0.0,
        highres_structure_skeleton_weight=0.0,
        use_legacy_stage_connectivity_loss=False,
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
        self.boundary_weight = boundary_weight
        self.boundary_radius = boundary_radius
        self.boundary_loss = BCEDiceLoss(dice_weight=1.0, bce_weight=1.0)
        self.skeleton_stage_weight = skeleton_stage_weight
        self.skeleton_stage_weights = tuple(float(w) for w in skeleton_stage_weights)
        if stage_structure_weights is None:
            stage_structure_weights = (0.0, 0.0, 0.004, 0.006)
        self.stage_structure_weights = tuple(float(w) for w in stage_structure_weights)
        self.stage_connectivity_factor = float(stage_connectivity_factor)
        self.stage_direction_factor = float(stage_direction_factor)
        self.stage_skeleton_connectivity_s2c_weight = float(
            stage_skeleton_connectivity_s2c_weight
        )
        self.stage_skeleton_connectivity_c2s_weight = float(
            stage_skeleton_connectivity_c2s_weight
        )
        self.use_legacy_stage_connectivity_loss = bool(
            use_legacy_stage_connectivity_loss
        )
        self.road_attention_weight = float(road_attention_weight)
        self.highres_structure_skeleton_weight = float(highres_structure_skeleton_weight)
        self.highres_hard_weight_gamma = 2.0
        self.highres_hard_weight_lambda = 2.0

    @staticmethod
    def _match_spatial_size(target, reference, mode="nearest"):
        if target is None or target.shape[-2:] == reference.shape[-2:]:
            return target
        return F.interpolate(
            target.float(),
            size=reference.shape[-2:],
            mode=mode,
        )

    @staticmethod
    def _spatial_boundary_mask(reference, valid_mask=None):
        batch, _, height, width = reference.shape
        mask = torch.ones(
            (batch, 1, height, width),
            device=reference.device,
            dtype=reference.dtype,
        )
        if height > 0:
            mask[:, :, 0, :] = 0
            mask[:, :, -1, :] = 0
        if width > 0:
            mask[:, :, :, 0] = 0
            mask[:, :, :, -1] = 0
        if valid_mask is not None:
            if valid_mask.dim() == 3:
                valid_mask = valid_mask.unsqueeze(1)
            valid_mask = valid_mask.to(device=reference.device, dtype=reference.dtype)
            if valid_mask.shape[-2:] != (height, width):
                valid_mask = F.interpolate(valid_mask, size=(height, width), mode="nearest")
            mask = mask * valid_mask
        return mask

    @staticmethod
    def _connectivity_boundary_mask(reference, valid_mask=None):
        batch, _, height, width = reference.shape
        mask = torch.ones(
            (batch, 8, height, width),
            device=reference.device,
            dtype=reference.dtype,
        )
        mask[:, 0, 0, :] = 0      # N
        mask[:, 1, -1, :] = 0     # S
        mask[:, 2, :, 0] = 0      # W
        mask[:, 3, :, -1] = 0     # E
        mask[:, 4, 0, :] = 0      # NW
        mask[:, 4, :, 0] = 0
        mask[:, 5, 0, :] = 0      # NE
        mask[:, 5, :, -1] = 0
        mask[:, 6, -1, :] = 0     # SW
        mask[:, 6, :, 0] = 0
        mask[:, 7, -1, :] = 0     # SE
        mask[:, 7, :, -1] = 0
        if valid_mask is not None:
            if valid_mask.dim() == 3:
                valid_mask = valid_mask.unsqueeze(1)
            valid_mask = valid_mask.to(device=reference.device, dtype=reference.dtype)
            if valid_mask.shape[-2:] != (height, width):
                valid_mask = F.interpolate(valid_mask, size=(height, width), mode="nearest")
            mask = mask * valid_mask.expand(-1, 8, -1, -1)
        return mask

    def skeleton_pixel_loss(self, skeleton_logits, skeleton_gt, skeleton_dilate_gt):
        skeleton_gt = self._match_spatial_size(skeleton_gt, skeleton_logits)
        skeleton_dilate_gt = self._match_spatial_size(skeleton_dilate_gt, skeleton_logits)
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
        return loss_skeleton, bce_skeleton, dice_skeleton

    def highres_skeleton_pixel_loss(
        self,
        skeleton_logits,
        skeleton_gt,
        skeleton_dilate_gt,
        surface_logits,
    ):
        skeleton_gt = self._match_spatial_size(skeleton_gt, skeleton_logits)
        skeleton_dilate_gt = self._match_spatial_size(skeleton_dilate_gt, skeleton_logits)
        surface_logits = self._match_spatial_size(
            surface_logits.detach(),
            skeleton_logits,
            mode="bilinear",
        )
        surface_prob_detached = torch.sigmoid(surface_logits).detach()
        if surface_prob_detached.requires_grad:
            raise RuntimeError("High-res hard skeleton weights must not backprop into surface logits.")

        raw_pos_weight = (
            1.0
            + self.highres_hard_weight_lambda
            * (1.0 - surface_prob_detached).pow(self.highres_hard_weight_gamma)
        )
        positive_mask = skeleton_gt > 0.5
        pixel_weight = torch.ones_like(raw_pos_weight)
        if positive_mask.any():
            mean_pos_weight = raw_pos_weight[positive_mask].mean().detach()
            normalized_pos_weight = raw_pos_weight / mean_pos_weight.clamp_min(1e-6)
            pixel_weight[positive_mask] = normalized_pos_weight[positive_mask]
        else:
            normalized_pos_weight = torch.ones_like(raw_pos_weight)

        skeleton_pos_weight = (
            self.skeleton_loss.pos_weight.to(skeleton_logits.device)
            if self.skeleton_loss.pos_weight is not None
            else None
        )
        bce_map = F.binary_cross_entropy_with_logits(
            skeleton_logits,
            skeleton_dilate_gt,
            pos_weight=skeleton_pos_weight,
            reduction="none",
        )
        bce_skeleton = (bce_map * pixel_weight).mean()
        dice_skeleton = self.skeleton_loss.dice(skeleton_logits, skeleton_gt)
        loss_skeleton = bce_skeleton + self.skeleton_loss.dice_weight * dice_skeleton

        stats = {
            "hard_positive_weight_mean": skeleton_gt.new_tensor(1.0),
            "hard_positive_weight_min": skeleton_gt.new_tensor(1.0),
            "hard_positive_weight_max": skeleton_gt.new_tensor(1.0),
            "hard_weight_p_lt_02": skeleton_gt.new_tensor(1.0),
            "hard_weight_p_02_05": skeleton_gt.new_tensor(1.0),
            "hard_weight_p_ge_05": skeleton_gt.new_tensor(1.0),
        }
        if positive_mask.any():
            pos_weights = pixel_weight[positive_mask].detach()
            pos_probs = surface_prob_detached[positive_mask]
            stats["hard_positive_weight_mean"] = pos_weights.mean()
            stats["hard_positive_weight_min"] = pos_weights.min()
            stats["hard_positive_weight_max"] = pos_weights.max()
            bins = (
                ("hard_weight_p_lt_02", pos_probs < 0.2),
                ("hard_weight_p_02_05", (pos_probs >= 0.2) & (pos_probs < 0.5)),
                ("hard_weight_p_ge_05", pos_probs >= 0.5),
            )
            for name, mask in bins:
                stats[name] = pos_weights[mask].mean() if mask.any() else skeleton_gt.new_tensor(float("nan"))
        return loss_skeleton, bce_skeleton, dice_skeleton, stats

    @staticmethod
    def _shift_map(x, dy, dx):
        _, _, height, width = x.shape
        pad_left = max(dx, 0)
        pad_right = max(-dx, 0)
        pad_top = max(dy, 0)
        pad_bottom = max(-dy, 0)
        padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        y0 = max(-dy, 0)
        x0 = max(-dx, 0)
        return padded[:, :, y0:y0 + height, x0:x0 + width]

    def stage_connectivity_loss(
        self,
        connectivity_logits,
        connectivity_gt,
        skeleton_dilate_gt,
        positive_weight=2.0,
        valid_mask=None,
    ):
        if self.use_legacy_stage_connectivity_loss:
            return F.binary_cross_entropy_with_logits(
                connectivity_logits,
                connectivity_gt,
            )

        corridor = skeleton_dilate_gt.clamp(0.0, 1.0)
        if corridor.sum() <= 0:
            corridor = torch.ones_like(corridor)
        valid = self._connectivity_boundary_mask(connectivity_logits, valid_mask)

        bce_map = F.binary_cross_entropy_with_logits(
            connectivity_logits,
            connectivity_gt,
            reduction="none",
        )
        sample_weight = corridor * (
            1.0 + connectivity_gt * (float(positive_weight) - 1.0)
        ) * valid
        loss_bce = (bce_map * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)

        conn_prob = torch.sigmoid(connectivity_logits)
        pred_strength = conn_prob.mean(dim=1, keepdim=True)
        target_strength = connectivity_gt.max(dim=1, keepdim=True).values
        edge_extractor = EdgeAwareLoss(edge_width=3)
        pred_edge = edge_extractor.extract_edge(pred_strength)
        target_edge = edge_extractor.extract_edge(target_strength)
        edge_intersection = (pred_edge * target_edge * corridor * valid).sum(dim=(1, 2, 3))
        edge_union = (
            (pred_edge * corridor * valid).sum(dim=(1, 2, 3))
            + (target_edge * corridor * valid).sum(dim=(1, 2, 3))
        )
        loss_edge_dice = (
            1.0 - (2.0 * edge_intersection + 1.0) / (edge_union + 1.0)
        ).mean()

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
        opposite = (1, 0, 3, 2, 7, 6, 5, 4)
        symmetry_terms = []
        for d_idx, (dy, dx) in enumerate(directions):
            forward = torch.sigmoid(connectivity_logits[:, d_idx:d_idx + 1])
            backward = torch.sigmoid(
                self._shift_map(
                    connectivity_logits[:, opposite[d_idx]:opposite[d_idx] + 1],
                    -dy,
                    -dx,
                )
            )
            symmetry_terms.append(
                (torch.abs(forward - backward) * corridor * valid[:, d_idx:d_idx + 1]).sum()
                / (corridor * valid[:, d_idx:d_idx + 1]).sum().clamp_min(1.0)
            )
        loss_symmetry = torch.stack(symmetry_terms).mean()

        return loss_bce + 0.30 * loss_edge_dice + 0.20 * loss_symmetry

    def stage_skeleton_loss(self, stage_outputs, skeleton_gt, skeleton_dilate_gt):
        if not stage_outputs:
            return skeleton_gt.sum() * 0.0

        total = skeleton_gt.sum() * 0.0
        for idx, stage_output in enumerate(stage_outputs):
            if idx >= len(self.skeleton_stage_weights):
                break
            stage_logits = stage_output["skeleton"]
            target_size = stage_logits.shape[-2:]
            stage_skel = F.interpolate(
                skeleton_gt,
                size=target_size,
                mode="nearest",
            )
            stage_skel_dilate = F.interpolate(
                skeleton_dilate_gt,
                size=target_size,
                mode="nearest",
            )
            loss_stage, _, _ = self.skeleton_pixel_loss(
                stage_logits,
                stage_skel,
                stage_skel_dilate,
            )
            total = total + self.skeleton_stage_weights[idx] * loss_stage

        return total

    @staticmethod
    def _connectivity_neighbor_skeleton(skeleton_prob):
        padded = F.pad(skeleton_prob, (1, 1, 1, 1))
        height, width = skeleton_prob.shape[-2:]
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
        neighbors = []
        for dy, dx in directions:
            y0 = 1 + dy
            x0 = 1 + dx
            neighbors.append(padded[:, :, y0:y0 + height, x0:x0 + width])
        return torch.cat(neighbors, dim=1)

    def skeleton_connectivity_consistency_loss(
        self,
        skeleton_logits,
        connectivity_logits,
    ):
        skeleton_prob = torch.sigmoid(skeleton_logits)
        connectivity_prob = torch.sigmoid(connectivity_logits)
        skeleton_pair = skeleton_prob * self._connectivity_neighbor_skeleton(
            skeleton_prob
        )
        loss_s_to_c = (skeleton_pair * (1.0 - connectivity_prob)).mean()
        loss_c_to_s = (connectivity_prob * (1.0 - skeleton_pair)).mean()
        return (
            self.stage_skeleton_connectivity_s2c_weight * loss_s_to_c
            + self.stage_skeleton_connectivity_c2s_weight * loss_c_to_s
        )

    def stage_structure_loss(
        self,
        stage_outputs,
        skeleton_gt,
        skeleton_dilate_gt,
        stage_skeleton_gt=None,
        stage_skeleton_dilate_gt=None,
        connectivity_gt=None,
        direction_gt=None,
        valid_mask=None,
    ):
        if not stage_outputs:
            return skeleton_gt.sum() * 0.0

        total = skeleton_gt.sum() * 0.0
        for idx, stage_output in enumerate(stage_outputs):
            if "skeleton" not in stage_output or "connectivity" not in stage_output:
                continue
            try:
                stage_index = int(stage_output.get("stage", idx))
            except (TypeError, ValueError):
                continue
            if stage_index >= len(self.stage_structure_weights):
                continue

            stage_weight = (
                self.stage_structure_weights[stage_index]
                * float(stage_output.get("stage_loss_scale", 1.0))
            )
            if stage_weight <= 0:
                continue

            stage_skeleton_logits = stage_output["skeleton"]
            stage_connectivity_logits = stage_output["connectivity"]
            target_size = stage_skeleton_logits.shape[-2:]
            source_stage_skel = (
                stage_skeleton_gt
                if stage_skeleton_gt is not None
                else skeleton_gt
            )
            source_stage_skel_dilate = (
                stage_skeleton_dilate_gt
                if stage_skeleton_dilate_gt is not None
                else skeleton_dilate_gt
            )
            if source_stage_skel.shape[-2:] == target_size:
                stage_skel = source_stage_skel
                stage_skel_dilate = source_stage_skel_dilate
            else:
                stage_skel = F.interpolate(
                    source_stage_skel,
                    size=target_size,
                    mode="nearest",
                )
                stage_skel_dilate = F.interpolate(
                    source_stage_skel_dilate,
                    size=target_size,
                    mode="nearest",
                )
            loss_skeleton_stage, _, _ = self.skeleton_pixel_loss(
                stage_skeleton_logits,
                stage_skel,
                stage_skel_dilate,
            )
            loss_connectivity_stage = loss_skeleton_stage * 0.0
            loss_skeleton_connectivity_consistency = loss_skeleton_stage * 0.0
            direction_logits = stage_output.get("direction")
            if direction_logits is not None and self.stage_direction_factor > 0:
                if direction_gt is None:
                    loss_direction_stage = self.direction_field_loss(
                        direction_logits,
                        stage_skel,
                    )
                else:
                    stage_direction_gt = F.interpolate(
                        direction_gt,
                        size=target_size,
                        mode="nearest",
                    )
                    stage_direction_gt = F.normalize(stage_direction_gt, dim=1, eps=1e-6)
                    stage_valid = stage_skel * self._spatial_boundary_mask(
                        stage_direction_gt,
                        valid_mask,
                    )
                    direction_pred = F.normalize(direction_logits, dim=1, eps=1e-6)
                    direction_cosine = (
                        direction_pred * stage_direction_gt
                    ).sum(dim=1, keepdim=True)
                    loss_direction_stage = (
                        (1.0 - direction_cosine) * stage_valid
                    ).sum() / stage_valid.sum().clamp_min(1.0)
            else:
                loss_direction_stage = loss_skeleton_stage * 0.0
            total = total + stage_weight * (
                loss_skeleton_stage
                + self.stage_direction_factor * loss_direction_stage
            )

        return total

    def build_direction_target(self, skeleton):
        skel = (skeleton > 0.5).to(dtype=skeleton.dtype)
        directions = [
            (0, 1, 1.0, 0.0),
            (1, 0, -1.0, 0.0),
            (1, 1, 0.0, 1.0),
            (1, -1, 0.0, -1.0),
        ]
        scores = []
        targets = []
        for dy, dx, cos2, sin2 in directions:
            forward = self._shift_map(skel, dy, dx)
            backward = self._shift_map(skel, -dy, -dx)
            support = forward + backward + 2.0 * forward * backward
            scores.append(support)
            targets.append(
                torch.cat(
                    [
                        torch.full_like(skel, cos2),
                        torch.full_like(skel, sin2),
                    ],
                    dim=1,
                )
            )
        score_stack = torch.cat(scores, dim=1)
        target_stack = torch.stack(targets, dim=1)
        index = score_stack.argmax(dim=1, keepdim=True)
        gather_index = index.unsqueeze(1).expand(-1, 1, 2, -1, -1)
        target = torch.gather(target_stack, dim=1, index=gather_index).squeeze(1)
        return target, skel

    def direction_field_loss(self, direction_logits, skeleton_gt):
        direction_target, skeleton_mask = self.build_direction_target(skeleton_gt)
        direction_pred = F.normalize(direction_logits, dim=1, eps=1e-6)
        direction_target = direction_target.to(
            device=direction_pred.device,
            dtype=direction_pred.dtype,
        )
        skeleton_mask = skeleton_mask.to(
            device=direction_pred.device,
            dtype=direction_pred.dtype,
        )
        cosine = (direction_pred * direction_target).sum(dim=1, keepdim=True)
        loss_map = (1.0 - cosine) * skeleton_mask
        return loss_map.sum() / skeleton_mask.sum().clamp_min(1.0)

    def road_attention_loss(self, stage_outputs, surface_gt):
        if not stage_outputs or self.road_attention_weight <= 0:
            return surface_gt.sum() * 0.0

        total = surface_gt.sum() * 0.0
        for stage_output in stage_outputs:
            road_attention = stage_output.get("road_attention")
            if road_attention is None:
                continue

            target_size = road_attention.shape[-2:]
            road_target = surface_gt
            if road_target.shape[-2:] != target_size:
                road_target = F.interpolate(
                    road_target,
                    size=target_size,
                    mode="nearest",
                )
            road_attention = road_attention.clamp(1e-6, 1.0 - 1e-6)
            bce = F.binary_cross_entropy(road_attention, road_target.float())
            intersection = (road_attention * road_target).sum(dim=(1, 2, 3))
            union = road_attention.sum(dim=(1, 2, 3)) + road_target.sum(dim=(1, 2, 3))
            dice = (1.0 - (2.0 * intersection + 1.0) / (union + 1.0)).mean()
            total = total + self.road_attention_weight * (bce + dice)
        return total

    def highres_structure_skeleton_loss(
        self,
        stage_outputs,
        skeleton_gt,
        skeleton_dilate_gt,
        surface_logits,
    ):
        if not stage_outputs or self.highres_structure_skeleton_weight <= 0:
            zero = skeleton_gt.sum() * 0.0
            stats = {
                "highres_structure_skeleton_raw": zero.detach(),
                "hard_positive_weight_mean": zero.detach(),
                "hard_positive_weight_min": zero.detach(),
                "hard_positive_weight_max": zero.detach(),
                "hard_weight_p_lt_02": zero.detach(),
                "hard_weight_p_02_05": zero.detach(),
                "hard_weight_p_ge_05": zero.detach(),
            }
            return zero, stats

        total = skeleton_gt.sum() * 0.0
        raw_total = skeleton_gt.sum() * 0.0
        stat_totals = {}
        stat_count = 0
        for stage_output in stage_outputs:
            skeleton_logits = stage_output.get("highres_structure_skeleton")
            if skeleton_logits is None:
                continue
            skeleton_logits_full = F.interpolate(
                skeleton_logits,
                size=skeleton_gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss_skeleton, _, _, stats = self.highres_skeleton_pixel_loss(
                skeleton_logits_full,
                skeleton_gt,
                skeleton_dilate_gt,
                surface_logits,
            )
            total = total + self.highres_structure_skeleton_weight * loss_skeleton
            raw_total = raw_total + loss_skeleton
            for key, value in stats.items():
                if torch.isfinite(value):
                    stat_totals[key] = stat_totals.get(key, value.new_tensor(0.0)) + value
            stat_count += 1
        if stat_count > 0:
            stats = {
                key: (value / stat_count).detach()
                for key, value in stat_totals.items()
            }
            stats["highres_structure_skeleton_raw"] = (raw_total / stat_count).detach()
            for key in (
                "hard_positive_weight_mean",
                "hard_positive_weight_min",
                "hard_positive_weight_max",
                "hard_weight_p_lt_02",
                "hard_weight_p_02_05",
                "hard_weight_p_ge_05",
            ):
                if key not in stats:
                    stats[key] = raw_total.new_tensor(float("nan")).detach()
        else:
            stats = {
                "highres_structure_skeleton_raw": raw_total.detach(),
                "hard_positive_weight_mean": raw_total.detach(),
                "hard_positive_weight_min": raw_total.detach(),
                "hard_positive_weight_max": raw_total.detach(),
                "hard_weight_p_lt_02": raw_total.detach(),
                "hard_weight_p_02_05": raw_total.detach(),
                "hard_weight_p_ge_05": raw_total.detach(),
            }
        return total, stats

    def forward(
        self,
        surface_logits,
        skeleton_logits=None,
        connectivity_logits=None,
        surface_gt=None,
        skeleton_gt=None,
        skeleton_dilate_gt=None,
        stage_outputs=None,
        boundary_logits=None,
        stage_skeleton_gt=None,
        stage_skeleton_dilate_gt=None,
        connectivity_gt=None,
        direction_gt=None,
        boundary_gt=None,
        valid_mask=None,
    ):
        if surface_gt is None or skeleton_gt is None:
            raise ValueError("surface_gt and skeleton_gt are required.")

        loss_surface, bce_surface, dice_surface = self.surface_loss(surface_logits, surface_gt)
        if skeleton_dilate_gt is None:
            skeleton_dilate_gt = skeleton_gt

        if skeleton_logits is not None:
            loss_skeleton, bce_skeleton, dice_skeleton = self.skeleton_pixel_loss(
                skeleton_logits,
                skeleton_gt,
                skeleton_dilate_gt,
            )
        else:
            loss_skeleton = surface_logits.sum() * 0.0
            bce_skeleton = loss_skeleton.detach()
            dice_skeleton = loss_skeleton.detach()

        if connectivity_logits is not None and self.connectivity_weight > 0:
            if connectivity_gt is None:
                connectivity_gt = build_connectivity_target(
                    skeleton_gt,
                    erode_kernel_size=self.connectivity_erode_kernel_size,
                )
            connectivity_gt = connectivity_gt.to(
                device=connectivity_logits.device,
                dtype=connectivity_logits.dtype,
            )
            connectivity_gt = self._match_spatial_size(connectivity_gt, connectivity_logits)
            connectivity_valid = self._connectivity_boundary_mask(connectivity_logits, valid_mask)
            connectivity_loss_map = F.binary_cross_entropy_with_logits(
                connectivity_logits,
                connectivity_gt,
                reduction="none",
            )
            loss_connectivity = (connectivity_loss_map * connectivity_valid).sum() / connectivity_valid.sum().clamp_min(1.0)
        else:
            loss_connectivity = surface_logits.sum() * 0.0
        if boundary_logits is not None and self.boundary_weight > 0:
            if boundary_gt is None:
                boundary_gt = build_boundary_target(
                    surface_gt,
                    radius=self.boundary_radius,
                )
            boundary_gt = boundary_gt.to(
                device=boundary_logits.device,
                dtype=boundary_logits.dtype,
            )
            boundary_gt = self._match_spatial_size(boundary_gt, boundary_logits)
            loss_boundary, bce_boundary, dice_boundary = self.boundary_loss(
                boundary_logits,
                boundary_gt,
            )
        else:
            loss_boundary = surface_logits.sum() * 0.0
            bce_boundary = loss_boundary.detach()
            dice_boundary = loss_boundary.detach()

        if self.skeleton_stage_weight > 0 and stage_outputs:
            loss_skeleton_stage = self.stage_skeleton_loss(
                stage_outputs,
                skeleton_gt,
                skeleton_dilate_gt,
            )
        else:
            loss_skeleton_stage = surface_logits.sum() * 0.0

        loss_stage_structure = self.stage_structure_loss(
            stage_outputs,
            skeleton_gt,
            skeleton_dilate_gt,
            stage_skeleton_gt=stage_skeleton_gt,
            stage_skeleton_dilate_gt=stage_skeleton_dilate_gt,
            connectivity_gt=connectivity_gt,
            direction_gt=direction_gt,
            valid_mask=valid_mask,
        )
        loss_road_attention = self.road_attention_loss(stage_outputs, surface_gt)
        loss_highres_structure_skeleton, highres_stats = self.highres_structure_skeleton_loss(
            stage_outputs,
            skeleton_gt,
            skeleton_dilate_gt,
            surface_logits,
        )

        total_loss = (
            loss_surface
            + self.skeleton_weight * loss_skeleton
            + self.connectivity_weight * loss_connectivity
            + self.boundary_weight * loss_boundary
            + self.skeleton_stage_weight * loss_skeleton_stage
            + loss_stage_structure
            + loss_road_attention
            + loss_highres_structure_skeleton
        )

        loss_dict = {
            "total_loss": total_loss.detach(),
            "surface_loss": loss_surface.detach(),
            "surface_total_loss": loss_surface.detach(),
            "skeleton_loss": loss_skeleton.detach(),
            "connectivity_loss": loss_connectivity.detach(),
            "boundary_loss": loss_boundary.detach(),
            "skeleton_stage_loss": loss_skeleton_stage.detach(),
            "stage_structure_loss": loss_stage_structure.detach(),
            "road_attention_loss": loss_road_attention.detach(),
            "loss_highres_structure_skeleton": loss_highres_structure_skeleton.detach(),
            "highres_structure_skeleton_raw": highres_stats["highres_structure_skeleton_raw"],
            "hard_positive_weight_mean": highres_stats["hard_positive_weight_mean"],
            "hard_positive_weight_min": highres_stats["hard_positive_weight_min"],
            "hard_positive_weight_max": highres_stats["hard_positive_weight_max"],
            "hard_weight_p_lt_02": highres_stats["hard_weight_p_lt_02"],
            "hard_weight_p_02_05": highres_stats["hard_weight_p_02_05"],
            "hard_weight_p_ge_05": highres_stats["hard_weight_p_ge_05"],
            "surface_bce": bce_surface,
            "surface_dice": dice_surface,
            "skeleton_bce": bce_skeleton,
            "skeleton_dice": dice_skeleton,
            "boundary_bce": bce_boundary,
            "boundary_dice": dice_boundary,
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
