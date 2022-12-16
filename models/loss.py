
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
  def __init__(self, alpha=0.25, gamma=1):
    super(FocalLoss, self).__init__()
    self.alpha = alpha
    self.gamma = gamma

  def forward(self, input, target):
    ce_loss = F.cross_entropy(input, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = ce_loss * self.alpha * (1-pt)**self.gamma
    return torch.mean(focal_loss)