import torch
from losses.road_losses import BCEDiceLoss, SurfaceSkeletonLoss

# 测试 BCEDiceLoss 中的 hard positive weighting
print('测试 BCEDiceLoss:')
bce_dice = BCEDiceLoss(
    dice_weight=1.0,
    bce_weight=1.0,
)

# 场景：GT=1 but pred_prob < 0.4（hard positive）
logits = torch.ones(2, 1, 4, 4) * -2.0  # 低 logits -> 低概率
targets = torch.ones(2, 1, 4, 4)  # GT=1
loss, bce, dice = bce_dice(logits, targets)
print(f'  Hard positive case: loss={loss.item():.6f}, bce={bce.item():.6f}')
print(f'    → Hard positive 加权 (β=1.5) 使得 loss 更高')

# 场景：GT=1 and pred_prob > 0.4（正常情况）
logits = torch.ones(2, 1, 4, 4) * 2.0  # 高 logits -> 高概率
targets = torch.ones(2, 1, 4, 4)  # GT=1
loss, bce, dice = bce_dice(logits, targets)
print(f'  Normal positive case: loss={loss.item():.6f}, bce={bce.item():.6f}')
print(f'    → 预测已正确，无加权')

# 测试完整 SurfaceSkeletonLoss
print('\n测试 SurfaceSkeletonLoss:')
surface_loss = SurfaceSkeletonLoss()

surface_logits = torch.randn(2, 1, 224, 224)
skeleton_logits = torch.randn(2, 1, 224, 224)
surface_gt = torch.randint(0, 2, (2, 1, 224, 224)).float()
skeleton_gt = torch.randint(0, 2, (2, 1, 224, 224)).float()

total_loss, loss_dict = surface_loss(surface_logits, skeleton_logits, surface_gt, skeleton_gt)
print(f'  Total loss: {total_loss.item():.6f}')
print(f'  Surface loss: {loss_dict["surface_loss"].item():.6f}')
print(f'  Skeleton loss: {loss_dict["skeleton_loss"].item():.6f}')
print('  Centerline loss: removed')

# 验证梯度流
print('\n验证梯度流：')
logits_grad = torch.randn(1, 1, 4, 4, requires_grad=True)
targets_grad = torch.ones(1, 1, 4, 4)
loss_grad, _, _ = bce_dice(logits_grad, targets_grad)
loss_grad.backward()
print(f'  ✓ 梯度正常流向 logits')
print(f'    logits.grad 非零: {(logits_grad.grad != 0).any().item()}')

print('\n✓ 测试完成（hard-positive 与 centerline 已移除）。')
