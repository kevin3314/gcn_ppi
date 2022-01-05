import torch
import torch.nn
from transformers import AutoModel


class BioBERTClassifier(torch.nn.Module):
    def __init__(self, pretrianed="dmis-lab/biobert-v1.1", with_lstm=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrianed)
        hidden_size = self.encoder.config.hidden_size
        self.with_lstm = with_lstm
        if with_lstm:
            self.lstm = torch.nn.LSTM(hidden_size, hidden_size, num_layers=3, batch_first=True, bidirectional=True)
            self.linear = torch.nn.Linear(hidden_size * 2, 1)
        else:
            self.linear = torch.nn.Linear(hidden_size, 1)
        self.dropout = torch.nn.Dropout(self.encoder.config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states = self.encoder(
            input_ids, token_type_ids, attention_mask, output_hidden_states=True, return_dict=True
        ).hidden_states
        hidden_states = torch.stack(hidden_states[-4:], dim=0).mean(dim=0)
        if self.with_lstm:
            lstm_out, (ht, ct) = self.lstm(hidden_states)
            hidden_states = self.dropout(lstm_out.mean(dim=-2))  # (b, seq, hid*2) -> (b, 1)
            # hidden_states = ht[-2:, :, :].permute(1, 2, 0).contiguous().view(batch_size, -1)
            hidden_states = self.linear(self.dropout(hidden_states))
        else:
            hidden_states = self.linear(self.dropout(hidden_states[:, 0, :]))  # (b, seq, hid) -> (b, 1)
        return torch.squeeze(hidden_states, dim=-1)
