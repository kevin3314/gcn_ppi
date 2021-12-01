from itertools import chain

import torch.nn
import torch_geometric.data


class GNNForProtein(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_gnn_layers=2):
        super(GNNForProtein, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=embedding_dim)
        self.gnn_layers = torch.nn.ModuleList(
            [torch_geometric.nn.GraphConv(embedding_dim, embedding_dim) for _ in range(num_gnn_layers)]
        )

    def forward(self, graph: torch_geometric.data.Data, amino_acids_numbers: torch.Tensor):  # (batch_size)
        nodes = self.embedding(graph.x)
        for gnn_layer in self.gnn_layers:
            nodes = gnn_layer(nodes, graph.edge_index)
        # (batch_size, embedding_dim)
        nodes: torch.Tensor = torch.stack(
            [nodes[start:end].mean(dim=0) for start, end in zip(chain([0], amino_acids_numbers), amino_acids_numbers)]
        )
        return nodes
