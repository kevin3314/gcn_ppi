import logging
from typing import Any, List

import torch
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CommonMixin:
    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        preds = torch.cat([x["preds"] for x in outputs], dim=0)
        targets = torch.cat([x["targets"] for x in outputs], dim=0)

        # `outputs` is a list of dicts returned from `training_step()`
        self.train_acc(preds, targets.long())
        self.train_prec(preds, targets.long())
        self.train_rec(preds, targets.long())
        self.train_f1(preds, targets.long())
        try:
            self.train_auroc(preds, targets.long())
            auroc = self.train_auroc.compute()
        # If there is no positive instance
        except (ValueError, IndexError):
            auroc = 0
        # log train metrics
        self.log("train/acc", self.train_acc, prog_bar=False)
        self.log("train/prec", self.train_prec, prog_bar=False)
        self.log("train/rec", self.train_rec, prog_bar=False)
        self.log("train/f1", self.train_f1, prog_bar=True)
        self.log("train/auroc", auroc, prog_bar=True)
        self.train_auroc.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        preds = torch.cat([x["preds"] for x in outputs], dim=0)
        targets = torch.cat([x["targets"] for x in outputs], dim=0)

        self.val_acc(preds, targets.long())
        self.val_prec(preds, targets.long())
        self.val_rec(preds, targets.long())
        self.val_f1(preds, targets.long())
        try:
            self.val_auroc(preds, targets.long())
            auroc = self.val_auroc.compute()
        # If there is no positive instance
        except (ValueError, IndexError):
            auroc = 0
        # log val metrics
        self.log("val/acc", self.val_acc, prog_bar=False)
        self.log("val/prec", self.val_prec, prog_bar=False)
        self.log("val/rec", self.val_rec, prog_bar=False)
        self.log("val/f1", self.val_f1, prog_bar=True)
        self.log("val/auroc", auroc, prog_bar=True)
        self.val_auroc.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        logger.info("test_epoch_end")
        preds = torch.cat([x["preds"] for x in outputs], dim=0)
        targets = torch.cat([x["targets"] for x in outputs], dim=0)

        self.test_acc(preds, targets.long())
        self.test_prec(preds, targets.long())
        self.test_rec(preds, targets.long())
        self.test_f1(preds, targets.long())
        try:
            self.test_auroc(preds, targets.long())
            auroc = self.test_auroc.compute()
        # If there is no positive instance
        except (ValueError, IndexError):
            auroc = 0
        # log test metrics
        self.log("test/acc", self.test_acc)
        self.log("test/prec", self.test_prec)
        self.log("test/rec", self.test_rec)
        self.log("test/f1", self.test_f1)
        self.log("test/auroc", auroc, prog_bar=True)
        self.test_auroc.reset()

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
