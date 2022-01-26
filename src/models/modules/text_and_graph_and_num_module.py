import logging

import torch.nn

from .graph_modality_model import GraphModalityModel
from .text_modality_model import TextModalityModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextAndGraphAndNumModel(torch.nn.Module):
    """Multimodal model of text and graph.
    Text module is BioBERT and graph module is graph neural network.
    """

    def __init__(
        self,
        amino_vocab_size: int,
        node_dim: int,
        num_gnn_layers: int,
        num_feature_dim: int,
        pretrained="dmis-lab/biobert-v1.1",
        with_lstm=False,
        with_intermediate_layer=False,
    ):
        super(TextAndGraphAndNumModel, self).__init__()
        self.text_model = TextModalityModel(pretrained, with_lstm)
        text_hidden_size = self.text_model.encoder.config.hidden_size
        text_hidden_size = text_hidden_size * (int(with_lstm) + 1)

        self.gnn = GraphModalityModel(amino_vocab_size, node_dim, num_gnn_layers)
        total_feature_dim = text_hidden_size + node_dim * 2 + num_feature_dim * 2
        if with_intermediate_layer:
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(total_feature_dim, total_feature_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(total_feature_dim // 2, 1),
            )
        else:
            self.linear = torch.nn.Linear(total_feature_dim, 1)
        self.dropout = torch.nn.Dropout(self.text_model.encoder.config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        data0,
        data1,
        numerical_features0,
        numerical_features1,
    ):
        text_emb = self.text_model(input_ids, token_type_ids, attention_mask)
        node0 = self.gnn(data0.x, data0.edge_index, data0.batch)  # (b, node_dim)
        node1 = self.gnn(data1.x, data1.edge_index, data1.batch)  # (b, node_dim)
        text_graph_hid = self.dropout(torch.cat([text_emb, node0, node1], dim=-1))  # (b, hid+2*node_dim)
        # logger.info(f"numerical_feature.shape: {numerical_features0.shape}")
        # logger.info(f"node.shape: {node0.shape}")
        hidden_state = torch.cat(
            [text_graph_hid, numerical_features0, numerical_features1], dim=-1
        )  # (b, hid+2*node_dim)
        # logger.info("hidden_state: {}".format(hidden_state.shape))
        logit = self.linear(hidden_state)
        return torch.squeeze(logit, dim=-1)
