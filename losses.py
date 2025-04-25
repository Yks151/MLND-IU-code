import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicFocalLoss(nn.Module):
    def __init__(self, gamma_base=2, beta=5, alpha=0.25):
        super().__init__()
        self.gamma_base = gamma_base
        self.beta = beta
        self.alpha = alpha

    def forward(self, pred, target):
        pred_prob = torch.sigmoid(pred)
        pt = torch.where(target == 1, pred_prob, 1 - pred_prob)
        gamma = self.gamma_base + self.beta * (1 - pt.detach())
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * (1 - pt) ** gamma

        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        return (focal_weight * bce_loss).mean()

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1 - (2 * intersection + self.eps) / (union + self.eps)

class MultiStageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = DynamicFocalLoss()
        self.dice = DiceLoss()

    def forward(self, outputs, targets):
        final_pred, aux_preds = outputs
        loss = 0

        # 主损失
        loss += self.focal(final_pred, targets)
        loss += self.dice(final_pred, targets)

        # 辅助损失（深度监督）
        for aux in aux_preds:
            loss += 0.5 * self.dice(aux, targets)

        return loss