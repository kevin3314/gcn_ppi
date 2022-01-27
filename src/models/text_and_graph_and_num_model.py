import logging
from typing import Any

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import F1, Precision, Recall

from src.models.mixin import CommonMixin
from src.models.modules.text_and_graph_and_num_module import TextAndGraphAndNumModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextAndGraphAndNumModule(CommonMixin, LightningModule):
    def __init__(
        self,
        amino_vocab_size: int,
        node_dim: int,
        num_gnn_layers: int,
        num_feature_dim: int,
        pretrained_path: str = "dmis-lab/biobert-v1.1",
        with_lstm=False,
        with_intermediate_layer=False,
        train_size: int = 358020,
        batch_size: int = 32,
        max_epochs: int = 50,
        lr: float = 5e-5,
        warmup_epoch: int = 5,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model: TextAndGraphAndNumModel = TextAndGraphAndNumModel(
            amino_vocab_size,
            node_dim,
            num_gnn_layers,
            num_feature_dim,
            pretrained_path,
            with_lstm,
            with_intermediate_layer,
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.train_prec = Precision()
        self.train_rec = Recall()
        self.train_f1 = F1()
        self.val_prec = Precision()
        self.val_rec = Recall()
        self.val_f1 = F1()
        self.test_prec = Precision()
        self.test_rec = Recall()
        self.test_f1 = F1()

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        data0,
        data1,
        numerical_feature0: torch.Tensor,
        numerical_feature1: torch.Tensor,
    ):
        return self.model(
            input_ids, token_type_ids, attention_mask, data0, data1, numerical_feature0, numerical_feature1
        )

    def step(self, batch: Any):
        input_ids, token_type_ids, attention_mask, data0, data1, numerical_feature0, numerical_feature1, labels = batch
        logits = self.forward(
            input_ids, token_type_ids, attention_mask, data0, data1, numerical_feature0, numerical_feature1
        )
        loss = self.criterion(logits, labels.float())
        preds = F.sigmoid(logits)
        # preds = torch.argmax(logits, dim=1)
        return loss, preds, labels
