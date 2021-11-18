from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy

from src.models.modules.graph_bert import GraphBertModelForNodeClassification
from src.models.modules.graph_bert_layers import GraphBertConfig


class GraphBertNodeClassificationModule(LightningModule):
    def __init__(
        self,
        config: GraphBertConfig,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model: GraphBertModelForNodeClassification = GraphBertModelForNodeClassification(config)
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(
        self,
        raw_features: torch.Tensor,
        wl_role_ids: torch.Tensor,
        init_pos_ids: torch.Tensor,
        hop_dis_ids: torch.Tensor,
    ):
        return self.model(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids)

    def step(self, batch: Any):
        raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, labels = batch
        logits = self.forward(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

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
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
