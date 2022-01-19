import logging

import torch.nn

from .text_modality_model import TextModalityModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextAndNumModel(torch.nn.Module):
    """Multimodal model of text and graph.
    Text module is BioBERT and graph module is graph neural network.
    """

    def __init__(
        self,
        num_feature_dim: int,
        pretrained="dmis-lab/biobert-v1.1",
        with_lstm=False,
    ):
        super(TextAndNumModel, self).__init__()
        self.text_model = TextModalityModel(pretrained, with_lstm)
        text_hidden_size = self.text_model.encoder.config.hidden_size
        text_hidden_size = text_hidden_size * (int(with_lstm) + 1)

        total_feature_dim = text_hidden_size + num_feature_dim * 2
        self.linear = torch.nn.Linear(total_feature_dim, 1)
        self.dropout = torch.nn.Dropout(self.text_model.encoder.config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_feature0: torch.Tensor,
        num_feature1: torch.Tensor,
    ):
        text_emb = self.dropout(self.text_model(input_ids, token_type_ids, attention_mask))
        hidden_state = torch.cat([text_emb, num_feature0, num_feature1], dim=-1)  # (b, hid+2*node_dim)
        logit = self.linear(self.dropout(hidden_state))
        return torch.squeeze(logit, dim=-1)
