import torch.nn

from .gnn_for_protein import GNNForProtein
from .graph_bert import GraphBertModel
from .graph_bert_layers import GraphBertConfig


class MultiModalModel(torch.nn.Module):
    """Multimodal Model for fusion of text modality and protein modality.
    At first, representation for each protein is calculated using GNN.
    Then two representation is added together and fed to GraphBert to obtain results.
    """

    def __init__(self, config: GraphBertConfig, amino_vocab_size: int, embedding_dim: int, num_gnn_layers: int):
        super(MultiModalModel, self).__init__()
        self.graph_bert = GraphBertModel(config)
        self.gnn = GNNForProtein(amino_vocab_size, embedding_dim, num_gnn_layers)

    def forward(
        self,
        raw_features: torch.Tensor,  # (batch_size, k, num_features)
        amino_acids_graph_data0,
        amino_acids_graph_data1,
        role_ids: torch.Tensor,  # (batch_size, k)
        position_ids: torch.Tensor,  # (batch_size, k)
        hop_ids: torch.Tensor,  # (batch_size, k)
    ):
        k = raw_features.shape[1]
        protein_nodes = []
        for i in range(k):
            nodes0 = getattr(amino_acids_graph_data0, f"x_{i}")
            edge0 = getattr(amino_acids_graph_data0, f"edge_{i}")
            nodes1 = getattr(amino_acids_graph_data1, f"x_{i}")
            edge1 = getattr(amino_acids_graph_data1, f"edge_{i}")
            batch_index0 = getattr(amino_acids_graph_data0, f"x_{i}_batch")
            batch_index1 = getattr(amino_acids_graph_data1, f"x_{i}_batch")
            nodes0 = self.gnn(nodes0, edge0, batch_index0)  # (b)
            nodes1 = self.gnn(nodes1, edge1, batch_index1)  # (b)
            nodes = nodes0 + nodes1
            protein_nodes.append(nodes)
        protein_nodes: torch.Tensor = torch.stack(protein_nodes, dim=0)  # (k, b, num_features)
        protein_nodes: torch.Tensor = protein_nodes.permute(1, 0, 2)  # (b, k, num_features)
        features = raw_features + protein_nodes
        return self.graph_bert(features, role_ids, position_ids, hop_ids)
