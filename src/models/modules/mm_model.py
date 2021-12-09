import logging

import torch.nn

from src.datamodules.datasets.graph_node_classification_dataset import NULL_EMBEDDING

from .gnn_for_protein import GNNForProtein
from .graph_bert import GraphBertModelForNodeClassification
from .graph_bert_layers import GraphBertConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MultiModalModel(torch.nn.Module):
    """Multimodal Model for fusion of text modality and protein modality.
    At first, representation for each protein is calculated using GNN.
    Then two representation is added together and fed to GraphBert to obtain results.
    """

    def __init__(self, config: GraphBertConfig, amino_vocab_size: int, embedding_dim: int, num_gnn_layers: int):
        super(MultiModalModel, self).__init__()
        self.graph_bert: GraphBertModelForNodeClassification = GraphBertModelForNodeClassification(config)
        self.gnn = GNNForProtein(amino_vocab_size, embedding_dim, num_gnn_layers)
        # Protein graph embedding in case pdb is not available.
        self.null_node = torch.nn.Parameter(torch.normal(0, 1, size=(1, embedding_dim)))

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
        # FIXME: Handle null embedding properly.
        # It is hard to replace null nodes with corresponding embedding because
        # batching mix several proteins together.
        for i in range(k):
            # If node is empty (i.e. torch.ones(1)), then use null embedding
            nodes0 = getattr(amino_acids_graph_data0, f"x_{i}")
            edge0 = getattr(amino_acids_graph_data0, f"edge_index_{i}")
            nodes1 = getattr(amino_acids_graph_data1, f"x_{i}")
            edge1 = getattr(amino_acids_graph_data1, f"edge_index_{i}")
            batch_index0 = getattr(amino_acids_graph_data0, f"x_{i}_batch")
            batch_index1 = getattr(amino_acids_graph_data1, f"x_{i}_batch")
            if nodes0.shape == NULL_EMBEDDING.shape and nodes0 == NULL_EMBEDDING.to(nodes0.device):
                nodes0 = self.null_node
            else:
                nodes0 = self.gnn(nodes0, edge0, batch_index0)  # (b)
            if nodes1.shape == NULL_EMBEDDING.shape and nodes1 == NULL_EMBEDDING.to(nodes0.device):
                nodes1 = self.null_node
            else:
                nodes1 = self.gnn(nodes1, edge1, batch_index1)  # (b)
            nodes = nodes0 + nodes1
            protein_nodes.append(nodes)
        protein_nodes: torch.Tensor = torch.stack(protein_nodes, dim=0)  # (k, b, p_num_features)
        protein_nodes: torch.Tensor = protein_nodes.permute(1, 0, 2)  # (b, k, p_num_features)
        features = torch.cat([raw_features, protein_nodes], dim=-1)  # (b, k, p_num_features + t_num_features)
        return self.graph_bert(features, role_ids, position_ids, hop_ids)
