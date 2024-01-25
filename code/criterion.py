import torch
import torch.nn as nn
import torch.nn.functional as F
from log import get_logger
import itertools


class CeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L0_lambda = 0.1

    def forward(self, output, target, reduction='mean'):
        criterion = nn.CrossEntropyLoss(reduction=reduction)
        criterion_mse = nn.MSELoss()
        logits_id, logits_gender, age_pred, logits_psy = output[0], output[1], output[2], output[3]
        id, gender, age, psy = target[0], target[1], target[2], target[3]
        loss = criterion(logits_id, id)
        # loss = criterion(logits_id, id) + criterion(logits_gender, gender) + criterion(logits_psy, psy)
        return loss