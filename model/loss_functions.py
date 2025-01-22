import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        assert(input.size() == target.size())

        pos = torch.eq(target, 1).float()
        neg = torch.eq(target, 0).float()
        num_pos = torch.sum(pos, dim=(2,3), keepdim=True)
        num_neg = torch.sum(neg, dim=(2,3), keepdim=True)
        num_total = num_pos + num_neg

        alpha = num_neg / num_total
        beta = num_pos / num_total
        weights = alpha * pos + beta * neg

        return F.binary_cross_entropy_with_logits(input, target, weights, reduction=self.reduction)


class DiceLoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(DiceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        assert(input.size() == target.size())

        input = torch.sigmoid(input)
        inter = ((input*target)).sum(dim=(2,3))
        union = ((input+target)).sum(dim=(2,3))
        dice = 1 - 2 * (inter+1) / (union+1)
        if self.reduction == 'mean':
            return dice.mean()
        elif self.reduction == 'sum':
            return dice.sum()

