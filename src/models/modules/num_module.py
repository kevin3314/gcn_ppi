import logging

import torch.nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NumModel(torch.nn.Module):
    """Monomodal model of numerical feature.
    Text module is based on BioBERT.
    """

    def __init__(self, num_feature_dim: int, dropout_prob: float = 0.1):
        super().__init__()
        self.linear = torch.nn.Linear(num_feature_dim, 1)

    def forward(self, feauture: torch.Tensor):
        logit = self.linear(feauture)
        return torch.squeeze(logit, dim=-1)
