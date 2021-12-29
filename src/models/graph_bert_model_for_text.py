import logging
from typing import Any, List

import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchmetrics import F1, Precision, Recall

from src.models.modules.graph_bert import GraphBertModelForNodeClassification
from src.models.modules.graph_bert_layers import GraphBertConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GraphBertNodeClassificationModuleForText(LightningModule):
    """This module works only with text modality."""

    def __init__(
        self,
        config: GraphBertConfig,
        lr: float = 1e-2,
        warmup_epoch: int = 5,
        eps: float = 1e-8,
        weight_decay: float = 5e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model: GraphBertModelForNodeClassification = GraphBertModelForNodeClassification(config)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
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
        raw_features: torch.Tensor,  # (batch_size, k, num_features)
        role_ids: torch.Tensor,  # (batch_size, k)
        position_ids: torch.Tensor,  # (batch_size, k)
        hop_ids: torch.Tensor,  # (batch_size, k)
    ):
        return self.model(raw_features, role_ids, position_ids, hop_ids)

    def step(self, batch: Any):
        raw_features, _, _, wl_role_ids, init_pos_ids, hop_dis_ids, labels = batch
        logits = self.forward(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids)
        loss = self.criterion(logits, labels.float())
        preds = F.sigmoid(logits)
        # preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        prec = self.train_prec(preds, targets.long())
        rec = self.train_rec(preds, targets.long())
        f1 = self.train_f1(preds, targets.long())
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/prec", prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/rec", rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        prec = self.val_prec(preds, targets.long())
        rec = self.val_rec(preds, targets.long())
        f1 = self.val_f1(preds, targets.long())
        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/prec", prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rec", rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        prec = self.test_prec(preds, targets.long())
        rec = self.test_rec(preds, targets.long())
        f1 = self.test_f1(preds, targets.long())

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/prec", prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/rec", rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
