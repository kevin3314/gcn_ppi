import logging

import torch.nn

from .graph_modality_model import GraphModalityModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GraphAndNumModel(torch.nn.Module):
    """Multimodal model of text and graph.
    Text module is BioBERT and graph module is graph neural network.
    """

    def __init__(
        self,
        amino_vocab_size: int,
        node_dim: int,
        num_gnn_layers: int,
        num_feature_dim: int,
        dropout_prob: float,
        pretrained="dmis-lab/biobert-v1.1",
        with_lstm=False,
    ):
        super(GraphAndNumModel, self).__init__()
        self.gnn = GraphModalityModel(amino_vocab_size, node_dim, num_gnn_layers)
        total_feature_dim = node_dim * 2 + num_feature_dim * 2
        self.linear = torch.nn.Linear(total_feature_dim, 1)
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(
        self,
        data0,
        data1,
        numerical_features0,
        numerical_features1,
    ):
        node0 = self.gnn(data0.x, data0.edge_index, data0.batch)  # (b, node_dim)
        node1 = self.gnn(data1.x, data1.edge_index, data1.batch)  # (b, node_dim)
        graph_hid = self.dropout(torch.cat([node0, node1], dim=-1))  # (b, hid+2*node_dim)
        hidden_state = torch.cat([graph_hid, numerical_features0, numerical_features1], dim=-1)  # (b, hid+2*node_dim)
        logit = self.linear(hidden_state)
        return torch.squeeze(logit, dim=-1)
