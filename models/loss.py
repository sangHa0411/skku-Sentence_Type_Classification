
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
  def __init__(self, alpha=0.25, gamma=1):
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma

  def forward(self, input, target):
    ce_loss = F.cross_entropy(input, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = ce_loss * self.alpha * (1-pt)**self.gamma
    return torch.mean(focal_loss)
	
class ArcFace(nn.Module):
	# Source from https://www.dacon.io/competitions/official/235875/codeshare/4589?page=1&dtype=recent
	def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.s = s
		self.m = m
		self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
		nn.init.xavier_uniform_(self.weight)
		
		self.easy_margin = easy_margin
		self.cos_m = math.cos(m)
		self.sin_m = math.sin(m)
		self.th = math.cos(math.pi - m)
		self.mm = math.sin(math.pi - m) * m
		
	def forward(self, input, label=None):
		cosine = F.linear(F.normalize(input), F.normalize(self.weight))
		sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
		phi = cosine * self.cos_m - sine * self.sin_m
		
		if self.easy_margin:
			phi = torch.where(cosine > 0, phi, cosine)
		else:
			phi = torch.where(cosine > self.th, phi, cosine - self.mm)
			
		if label is not None:
			one_hot = torch.zeros(cosine.size(), device='cuda')
			one_hot.scatter_(1, label.view(-1, 1).long(), 1)
			output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
		else:
			output = cosine
			
		output *= self.s
		return output