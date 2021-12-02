import logging

import torch.nn
import torch_geometric.data
import torch_geometric.nn
from torch_geometric.nn import global_mean_pool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GNNForProtein(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_gnn_layers=2):
        super(GNNForProtein, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=embedding_dim)
        self.gnn_layers = torch.nn.ModuleList(
            [torch_geometric.nn.GCNConv(embedding_dim, embedding_dim) for _ in range(num_gnn_layers)]
        )

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor, batch_index: torch.Tensor):  # (batch_size)
        nodes: torch.Tensor = self.embedding(nodes.long())
        for gnn_layer in self.gnn_layers:
            nodes = gnn_layer(nodes, edge_index).relu()
        # (batch_size, embedding_dim)
        nodes: torch.Tensor = global_mean_pool(nodes, batch_index)
        return nodes
