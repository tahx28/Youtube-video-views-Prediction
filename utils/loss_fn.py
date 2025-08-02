import torch
from torch import nn
import torch.nn.functional as F


class RMSLELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSLELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # Ensure the predictions and targets are non-negative
        # -- added a clamp to avoid log(0) and ReLU in the
        y_pred = torch.clamp(y_pred, min=0)
        y_true = torch.clamp(y_true, min=0)

        # Compute the RMSLE
        log_pred = torch.log1p(y_pred)
        log_true = torch.log1p(y_true)
        loss = torch.sqrt(torch.mean((log_pred - log_true) ** 2))

        return loss


class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Ensure the predictions and targets are non-negative
        # -- added a clamp to avoid log(0) and ReLU in the
        y_pred = torch.clamp(y_pred, min=0)
        y_true = torch.clamp(y_true, min=0)

        # Compute the MSLE
        log_pred = torch.log1p(y_pred)
        log_true = torch.log1p(y_true)
        loss = torch.mean((log_pred - log_true) ** 2)

        return loss

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self, preds, targets):
        return torch.sqrt(nn.functional.mse_loss(preds, targets))

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        abs_error = torch.abs(error)
        quadratic = torch.minimum(abs_error, torch.tensor(self.delta).to(y_pred.device))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return torch.mean(loss)



class CrossEntropyLossWrapper(nn.Module):
    def __init__(self):
        super(CrossEntropyLossWrapper, self).__init__()

    def forward(self, logits, targets):

        return F.cross_entropy(logits, targets)



class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha.to(logits.device) if self.alpha is not None else None)
        pt = torch.exp(-ce_loss)  
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss)

