import logging
from typing import Any, List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import F1, Precision, Recall
from transformers import AdamW, get_linear_schedule_with_warmup

from src.models.modules.graph_module import GraphModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GraphModule(LightningModule):
    """This module works only with text modality."""

    def __init__(
        self,
        amino_vocab_size: int,
        node_dim: int,
        num_gnn_layers: int,
        dropout_prob: float,
        train_size: int = 2326,
        batch_size: int = 32,
        max_epochs: int = 50,
        lr: float = 5e-5,
        warmup_epoch: int = 5,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model: GraphModel = GraphModel(amino_vocab_size, node_dim, num_gnn_layers, dropout_prob)
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

    def forward(self, data0, data1):
        return self.model(data0, data1)

    def step(self, batch: Any):
        data0, data1, labels = batch
        logits = self.forward(data0, data1)
        loss = self.criterion(logits, labels.float())
        preds = F.sigmoid(logits)
        # preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        preds = torch.cat([x["preds"] for x in outputs], dim=0)
        targets = torch.cat([x["targets"] for x in outputs], dim=0)

        # `outputs` is a list of dicts returned from `training_step()`
        self.train_prec(preds, targets.long())
        prec = self.train_prec.compute()
        self.train_rec(preds, targets.long())
        rec = self.train_rec.compute()
        self.train_f1(preds, targets.long())
        f1 = self.train_f1.compute()
        # log train metrics
        self.log("train/prec", prec, prog_bar=True)
        self.log("train/rec", rec, prog_bar=True)
        self.log("train/f1", f1, prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        preds = torch.cat([x["preds"] for x in outputs], dim=0)
        targets = torch.cat([x["targets"] for x in outputs], dim=0)

        self.val_prec(preds, targets.long())
        prec = self.val_prec.compute()
        self.val_rec(preds, targets.long())
        rec = self.val_rec.compute()
        self.val_f1(preds, targets.long())
        f1 = self.val_f1.compute()
        # log val metrics
        self.log("val/prec", prec, prog_bar=True)
        self.log("val/rec", rec, prog_bar=True)
        self.log("val/f1", f1, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        preds = torch.cat([x["preds"] for x in outputs], dim=0)
        targets = torch.cat([x["targets"] for x in outputs], dim=0)

        self.test_prec(preds, targets.long())
        prec = self.test_prec.compute()
        self.test_rec(preds, targets.long())
        rec = self.test_rec.compute()
        self.test_f1(preds, targets.long())
        f1 = self.test_f1.compute()
        # log test metrics
        self.log("test/prec", prec)
        self.log("test/rec", rec)
        self.log("test/f1", f1)

    def configure_optimizers(self):
        trainable_named_params = filter(lambda x: x[1].requires_grad, self.model.named_parameters())
        no_decay = ("bias", "LayerNorm.weight")
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in trainable_named_params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {"params": [p for n, p in trainable_named_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(params=optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.eps)
        steps_per_epoch = self.hparams.train_size // self.hparams.batch_size
        num_warmup_steps = self.hparams.warmup_epoch * steps_per_epoch
        num_training_steps = steps_per_epoch * self.trainer.max_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            ],
        )
