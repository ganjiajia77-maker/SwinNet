import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """
    Differentiable soft erosion for 2D binary probability maps.

    Args:
        img: [B, 1, H, W], values in [0, 1]

    Returns:
        eroded image with the same shape.
    """
    if img.dim() != 4:
        raise ValueError(f"Expected 4D tensor [B, C, H, W], got {img.shape}")

    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """
    Differentiable soft dilation for 2D binary probability maps.
    """
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    """
    Differentiable soft opening (erosion followed by dilation).
    """
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iter_num: int = 10) -> torch.Tensor:
    """
    Differentiable soft skeletonization.

    Args:
        img: [B, 1, H, W], probability map in [0, 1]
        iter_num: number of skeletonization iterations

    Returns:
        soft skeleton map, [B, 1, H, W]
    """
    img = img.clamp(0.0, 1.0)

    opened = soft_open(img)
    skeleton = F.relu(img - opened)

    for _ in range(iter_num):
        img = soft_erode(img)
        opened = soft_open(img)
        delta = F.relu(img - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)

    return skeleton.clamp(0.0, 1.0)


class SoftCLDiceLoss(nn.Module):
    """
    Soft clDice loss for binary road/vessel segmentation.

    This version is fully PyTorch-based and differentiable.
    It does not use scipy/skimage/cv2 skeletonization during training.
    
    Reference:
        Shit et al. "clDice - a Novel Topology-Preserving Loss Function for 
        Tubular Structure Segmentation", CVPR 2021
    """

    def __init__(self, iter_num: int = 10, smooth: float = 1.0):
        """
        Args:
            iter_num: number of soft skeletonization iterations
            smooth: Laplace smoothing constant
        """
        super().__init__()
        self.iter_num = iter_num
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute soft clDice loss.

        Args:
            logits:  [B, C, H, W] or [B, 1, H, W], raw model output
            targets: [B, 1, H, W] or [B, H, W], binary 0/1

        Returns:
            clDice loss scalar (1 - clDice).
        """
        # Ensure 4D tensors
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        # If logits is 2-class, take class 1
        if logits.shape[1] == 2:
            logits = logits[:, 1:2]
        
        # Convert to soft probabilities
        probs = torch.sigmoid(logits).clamp(0.0, 1.0)
        targets = targets.float().clamp(0.0, 1.0)

        # Compute soft skeletons
        pred_skel = soft_skeletonize(probs, iter_num=self.iter_num)
        target_skel = soft_skeletonize(targets, iter_num=self.iter_num)

        # Topological precision: predicted skeleton should lie inside GT mask.
        tprec = (
            torch.sum(pred_skel * targets, dim=(1, 2, 3)) + self.smooth
        ) / (
            torch.sum(pred_skel, dim=(1, 2, 3)) + self.smooth
        )

        # Topological sensitivity: GT skeleton should lie inside predicted mask.
        tsens = (
            torch.sum(target_skel * probs, dim=(1, 2, 3)) + self.smooth
        ) / (
            torch.sum(target_skel, dim=(1, 2, 3)) + self.smooth
        )

        # clDice coefficient
        cl_dice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-8)
        
        # Loss: 1 - clDice
        loss = 1.0 - cl_dice

        return loss.mean()


# Alias for backward compatibility
CLDiceLoss = SoftCLDiceLoss
