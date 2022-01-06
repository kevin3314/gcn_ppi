import logging

import torch.nn
from transformers import AutoModel

from .gnn_for_protein import GNNForProtein

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextAndGraphModel(torch.nn.Module):
    """Multimodal model of text and graph.
    Text module is BioBERT and graph module is graph neural network.
    """

    def __init__(self, amino_vocab_size: int, node_dim: int, num_gnn_layers: int, pretrained="dmis-lab/biobert-v1.1"):
        super(TextAndGraphModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained)
        hidden_size = self.encoder.config.hidden_size
        self.gnn = GNNForProtein(amino_vocab_size, node_dim, num_gnn_layers)
        # Protein graph embedding in case pdb is not available.
        self.null_node = torch.nn.Parameter(torch.normal(0, 1, size=(1, node_dim)))
        total_feature_dim = hidden_size + node_dim * 2
        self.linear = torch.nn.Linear(total_feature_dim, 1)
        self.dropout = torch.nn.Dropout(self.encoder.config.hidden_dropout_prob)

    def forward(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor, data0, data1
    ):
        node0 = self.gnn(data0.x, data0.edge_index, data0.batch)  # (b, node_dim)
        node1 = self.gnn(data1.x, data1.edge_index, data1.batch)  # (b, node_dim)
        text_emb = self.encoder(
            input_ids, token_type_ids, attention_mask, output_hidden_states=True, return_dict=True
        ).hidden_states[-1][
            :, 0, :
        ]  # (b, hid)
        hidden_state = torch.cat([text_emb, node0, node1], dim=-1)  # (b, hid+2*node_dim)
        logit = self.linear(self.dropout(hidden_state))
        return torch.squeeze(logit, dim=-1)
