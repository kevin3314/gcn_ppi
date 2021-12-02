import torch
from transformers.models.bert.modeling_bert import BertPooler, BertPreTrainedModel

from .graph_bert_layers import GraphBertEmbeddings, GraphBertEncoder

BertLayerNorm = torch.nn.LayerNorm


class GraphBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(GraphBertModel, self).__init__(config)
        self.config = config

        self.embeddings = GraphBertEmbeddings(config)
        self.encoder = GraphBertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.raw_feature_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.raw_feature_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, head_mask=None, residual_h=None):
        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            raw_features=raw_features, wl_role_ids=wl_role_ids, init_pos_ids=init_pos_ids, hop_dis_ids=hop_dis_ids
        )
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask, residual_h=residual_h)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        outputs = (
            sequence_output,
            pooled_output,
        ) + encoder_outputs[1:]
        return outputs
