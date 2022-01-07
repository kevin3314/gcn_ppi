import torch
import torch.nn

from .text_modality_model import TextModalityModel


class BioBERTClassifier(torch.nn.Module):
    def __init__(self, pretrianed="dmis-lab/biobert-v1.1", with_lstm=False):
        super().__init__()
        self.text_modality_model = TextModalityModel(pretrianed, with_lstm)
        hidden_size = self.text_modality_model.encoder.config.hidden_size
        if with_lstm:
            self.linear = torch.nn.Linear(hidden_size * 2, 1)
        else:
            self.linear = torch.nn.Linear(hidden_size, 1)
        self.dropout = torch.nn.Dropout(self.text_modality_model.encoder.config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states = self.text_modality_model(
            input_ids,
            token_type_ids,
            attention_mask,
        )
        logits = self.linear(self.dropout(hidden_states))
        return torch.squeeze(logits, dim=-1)
