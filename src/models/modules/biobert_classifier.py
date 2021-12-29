import torch
import torch.nn
from transformers import AutoModel


class BioBERTClassifier(torch.nn.Module):
    def __init__(self, pretrianed="dmis-lab/biobert-v1.1"):
        self.encoder = AutoModel.from_pretrained(pretrianed)
        hidden_size = self.encoder.config.hidden_size
        self.linear = torch.nn.Linear(hidden_size, 1)
        self.dropout = torch.nn.Dropout(self.encoder.config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states, _ = self.encoder(input_ids, token_type_ids, attention_mask)
        return self.linear(self.dropout(hidden_states[:, 0]))
