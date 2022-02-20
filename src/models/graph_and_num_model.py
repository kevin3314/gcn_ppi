import logging
from typing import Any

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import AUROC, F1, Accuracy, Precision, Recall

from src.models.mixin import CommonMixin
from src.models.modules.graph_and_num_module import GraphAndNumModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GraphAndNumModule(CommonMixin, LightningModule):
    def __init__(
        self,
        amino_vocab_size: int,
        node_dim: int,
        num_gnn_layers: int,
        num_feature_dim: int,
        dropout_prob: float,
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

        self.model: GraphAndNumModel = GraphAndNumModel(
            amino_vocab_size, node_dim, num_gnn_layers, num_feature_dim, dropout_prob
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.train_prec = Precision()
        self.train_rec = Recall()
        self.train_f1 = F1()
        self.train_auroc = AUROC()
        self.val_acc = Accuracy()
        self.val_prec = Precision()
        self.val_rec = Recall()
        self.val_f1 = F1()
        self.val_auroc = AUROC()
        self.test_acc = Accuracy()
        self.test_prec = Precision()
        self.test_rec = Recall()
        self.test_f1 = F1()
        self.test_auroc = AUROC()

    def forward(self, data0, data1, num_feature0: torch.Tensor, num_feature1: torch.Tensor):
        return self.model(data0, data1, num_feature0, num_feature1)

    def step(self, batch: Any):
        data0, data1, num_feature0, num_feature1, labels = batch
        logits = self.forward(data0, data1, num_feature0, num_feature1)
        loss = self.criterion(logits, labels.float())
        preds = F.sigmoid(logits)
        # preds = torch.argmax(logits, dim=1)
        return loss, preds, labels
