import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits[:, 1:2])                 # (B,1,H,W)
        tgt = (targets > 0).float().unsqueeze(1)              # (B,1,H,W)
        inter = (probs * tgt).sum(dim=(2,3))
        denom = (probs + tgt).sum(dim=(2,3))
        dice = (2*inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, targets):
        with torch.no_grad():
            t_bin = (targets > 0).to(torch.uint8).cpu().numpy()
            dist_maps = [distance_transform_edt(1 - t) for t in t_bin]
            dist = torch.from_numpy(np.stack(dist_maps,0)).to(logits.device).float()  # (B,H,W)
            w = 1.0 / (1.0 + dist)
        fg_logits = logits[:, 1]                    # (B,H,W)
        tgt = (targets > 0).float()                 # (B,H,W)
        return F.binary_cross_entropy_with_logits(fg_logits, tgt, weight=w, reduction='mean')

class BCEWithLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, logits, targets):
        fg_logits = logits[:, 1:2]                 # (B,1,H,W)
        tgt = (targets > 0).float().unsqueeze(1)   # (B,1,H,W)
        return self.loss(fg_logits, tgt)

class ComboLoss(nn.Module):
    def __init__(self, w_bce=1.0, w_dice=1.0, w_boundary=0.5):
        super().__init__()
        self.bce = BCEWithLogits()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
        self.w_bce = w_bce; self.w_dice = w_dice; self.w_boundary = w_boundary
    def forward(self, logits, targets):
        return (self.w_bce * self.bce(logits, targets)
              + self.w_dice * self.dice(logits, targets)
              + self.w_boundary * self.boundary(logits, targets))
