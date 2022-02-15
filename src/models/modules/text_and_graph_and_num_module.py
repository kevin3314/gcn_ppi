import logging
from typing import Optional

from torch import Tensor
import torch.nn
import torch.nn.functional as F

from .graph_modality_model import GraphModalityModel
from .text_modality_model import TextModalityModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TensorFusionNetwork(torch.nn.Module):
    """Tensor Fusion Network for amalgamation of multimodal information.
    Paper: Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." EMNLP 2017 Oral.
    Implementation comes from https://github.com/Justin1904/TensorFusionNetworks"""
    def __init__(self, text_dim: int, graph_dim: int, num_dim: int, condense_text_dim: int, condense_graph_dim: int, condense_num_dim: int, post_fusion_dim: int, post_fusion_dropout_prob: float):
        super().__init__()
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        self.num_dim = num_dim
        self.condense_text_dim = condense_text_dim
        self.condense_graph_dim = condense_graph_dim
        self.condense_num_dim = condense_num_dim
        self.post_fusion_dim = post_fusion_dim
        self.post_fusion_dropout_prob = post_fusion_dropout_prob

        self.text_condense_layer = torch.nn.Linear(self.text_dim, self.condense_text_dim)
        self.graph_condense_layer = torch.nn.Linear(self.graph_dim, self.condense_graph_dim)
        self.num_condense_layer = torch.nn.Linear(self.num_dim, self.condense_num_dim)
        self.post_fusion_dropout = torch.nn.Dropout(p=self.post_fusion_dropout_prob)
        self.post_fusion_layer_1 = torch.nn.Linear((self.condense_text_dim + 1) * (self.condense_graph_dim + 1) * (self.condense_num_dim+ 1), self.post_fusion_dim)
        self.post_fusion_layer_2 = torch.nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = torch.nn.Linear(self.post_fusion_dim, 1)

    def forward(self, text_x: Tensor, graph_x: Tensor, num_x: Tensor):
        """Foward based on xxx.

        Args:
            text_x (Tensor): Text data with (batch_size, text_dim)
            graph_x (Tensor): Graph data with (batch_size, graph_dim)
            num_x (Tensor): Numerical data with (batch_size, num_dim)
        """
        batch_size = text_x.shape[0]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product

        tmp = torch.ones(batch_size, 1).to(text_x.dtype).to(text_x.device)
        tmp.requires_grad = False
        _text_x = torch.cat([tmp, self.text_condense_layer(text_x)], dim=1)
        tmp = torch.ones(batch_size, 1).to(graph_x.dtype).to(graph_x.device)
        tmp.requires_grad = False
        _graph_x = torch.cat([tmp, self.graph_condense_layer(graph_x)], dim=1)
        tmp = torch.ones(batch_size, 1).to(num_x.dtype).to(num_x.device)
        tmp.requires_grad = False
        _num_x = torch.cat([tmp, self.num_condense_layer(num_x)], dim=1)

        # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
        # we want to perform outer product between the two batch, hence we unsqueenze them to get
        # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
        # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
        fusion_tensor = torch.bmm(_graph_x.unsqueeze(2), _num_x.unsqueeze(1))

        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it
        fusion_tensor = fusion_tensor.view(-1, (self.condense_graph_dim + 1) * (self.condense_num_dim + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_x.unsqueeze(1)).view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
        output = self.post_fusion_layer_3(post_fusion_y_2)

        return output

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
        with_tensorfusion_network=False,
        with_intermediate_layer=False,
        post_fusion_dim: Optional[int] = None,
        post_fusion_dropout_prob: Optional[float] = None,
        condense_text_dim: Optional[int] = None,
        condense_graph_dim: Optional[int] = None,
        condense_num_dim: Optional[int] = None,
    ):
        super(TextAndGraphAndNumModel, self).__init__()
        self.text_model = TextModalityModel(pretrained, with_lstm)
        text_hidden_size = self.text_model.encoder.config.hidden_size
        text_hidden_size = text_hidden_size * (int(with_lstm) + 1)

        self.gnn = GraphModalityModel(amino_vocab_size, node_dim, num_gnn_layers)
        total_feature_dim = text_hidden_size + node_dim * 2 + num_feature_dim * 2
        self.with_tensorfusion_network = with_tensorfusion_network
        if with_tensorfusion_network:
            assert post_fusion_dim is not None
            assert post_fusion_dropout_prob is not None
            self.post_model = TensorFusionNetwork(
                text_hidden_size,
                node_dim * 2,
                num_feature_dim * 2,
                condense_text_dim,
                condense_graph_dim,
                condense_num_dim,
                post_fusion_dim,
                post_fusion_dropout_prob,
            )
        elif with_intermediate_layer:
            self.post_model = torch.nn.Sequential(
                torch.nn.Linear(total_feature_dim, total_feature_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(total_feature_dim // 2, 1),
            )
        else:
            self.post_model = torch.nn.Linear(total_feature_dim, 1)
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
        if self.with_tensorfusion_network:
            text_hid = self.dropout(text_emb)
            graph_hid = self.dropout(torch.cat([node0, node1], dim=-1))
            num_hid = torch.cat([numerical_features0, numerical_features1], dim=-1)
            logit = self.post_model(text_hid, graph_hid, num_hid)
        else:
            text_graph_hid = self.dropout(torch.cat([text_emb, node0, node1], dim=-1))  # (b, hid+2*node_dim)
            hidden_state = torch.cat(
                [text_graph_hid, numerical_features0, numerical_features1], dim=-1
            )  # (b, hid+2*node_dim)
            # logger.info("hidden_state: {}".format(hidden_state.shape))
            logit = self.post_model(hidden_state)
        return torch.squeeze(logit, dim=-1)
