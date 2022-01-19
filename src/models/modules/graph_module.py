import logging

import torch.nn

from .graph_modality_model import GraphModalityModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GraphModel(torch.nn.Module):
    """Monomodal model of graph.
    Graph neural network is dedicated for graph module.
    """

    def __init__(
        self,
        amino_vocab_size: int,
        node_dim: int,
        num_gnn_layers: int,
        dropout_prob: float,
    ):
        super(GraphModel, self).__init__()
        self.gnn = GraphModalityModel(amino_vocab_size, node_dim, num_gnn_layers)
        self.linear = torch.nn.Linear(node_dim * 2, 1)
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, data0, data1):
        node0 = self.gnn(data0.x, data0.edge_index, data0.batch)  # (b, node_dim)
        node1 = self.gnn(data1.x, data1.edge_index, data1.batch)  # (b, node_dim)
        hidden_state = torch.cat([node0, node1], dim=-1)  # (b, 2*node_dim)
        logit = self.linear(self.dropout(hidden_state))
        return torch.squeeze(logit, dim=-1)
