import logging

import torch
import torch.nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NumModel(torch.nn.Module):
    """Monomodal model of numerical feature.
    Text module is based on BioBERT.
    """

    def __init__(self, num_feature_dim: int, dropout_prob: float = 0.1):
        super().__init__()
        self.linear1 = torch.nn.Linear(num_feature_dim * 2, num_feature_dim)
        self.linear2 = torch.nn.Linear(num_feature_dim, 1)
        # self.linear = torch.nn.Linear(num_feature_dim * 2, 1)

    def forward(self, num_feature0: torch.Tensor, num_feature1: torch.Tensor):
        hidden = F.relu(self.linear1(torch.cat([num_feature0, num_feature1], dim=-1)))
        return torch.squeeze(self.linear2(hidden), dim=-1)
        # return torch.squeeze(self.linear(torch.cat([num_feature0, num_feature1], dim=-1)), dim=-1)
