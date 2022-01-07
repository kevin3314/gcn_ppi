import logging

import torch.nn

from .text_modality_model import TextModalityModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextModel(torch.nn.Module):
    """Monomodal model of text.
    Text module is based on BioBERT.
    """

    def __init__(self, pretrained="dmis-lab/biobert-v1.1", with_lstm=False):
        super().__init__()
        self.text_model = TextModalityModel(pretrained, with_lstm)
        hidden_size = self.text_model.encoder.config.hidden_size
        hidden_size = hidden_size * (int(with_lstm) + 1)
        self.linear = torch.nn.Linear(hidden_size, 1)
        self.dropout = torch.nn.Dropout(self.text_model.encoder.config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states = self.text_model(input_ids, token_type_ids, attention_mask)
        logit = self.linear(self.dropout(hidden_states))
        return torch.squeeze(logit, dim=-1)
