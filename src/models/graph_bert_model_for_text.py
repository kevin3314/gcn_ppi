from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from transformers import AdamW, get_linear_schedule_with_warmup

from src.models.modules.graph_bert import GraphBertModelForNodeClassification
from src.models.modules.graph_bert_layers import GraphBertConfig


class GraphBertNodeClassificationModuleForText(LightningModule):
    """This module works only with text modality."""

    def __init__(
        self,
        config: GraphBertConfig,
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

        self.model: GraphBertModelForNodeClassification = GraphBertModelForNodeClassification(config)
        self.criterion = torch.nn.BCEWithLogitsLoss()

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
        loss = self.criterion(logits, labels.float())
        preds = (logits > 0.0).long()
        # preds = torch.argmax(logits, dim=1)
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
