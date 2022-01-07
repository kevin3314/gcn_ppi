import torch
import torch.nn
from transformers import AutoModel


class TextModalityModel(torch.nn.Module):
    def __init__(self, pretrianed="dmis-lab/biobert-v1.1", with_lstm=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrianed)
        self.with_lstm = with_lstm

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states = self.encoder(
            input_ids, token_type_ids, attention_mask, output_hidden_states=True, return_dict=True
        ).hidden_states
        hidden_states = torch.stack(hidden_states[-4:], dim=0).mean(dim=0)  # (b, seq, hid)
        if self.with_lstm:
            lstm_out, (ht, ct) = self.lstm(hidden_states)
            hidden_states = lstm_out.mean(dim=-2)  # (b, seq, hid*2) -> (b, hid*2)
            # hidden_states = ht[-2:, :, :].permute(1, 2, 0).contiguous().view(batch_size, -1)
        else:
            hidden_states = hidden_states[:, 0, :]  # (b, seq, hid) -> (b, hid)
        return hidden_states
