import logging
from typing import Any, List

import torch
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CommonMixin:
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
        try:
            self.train_auroc(preds, targets.long())
            auroc = self.train_auroc.compute()
        # If there is no positive instance
        except ValueError:
            auroc = 0
        # log train metrics
        self.log("train/prec", prec, prog_bar=True)
        self.log("train/rec", rec, prog_bar=True)
        self.log("train/f1", f1, prog_bar=True)
        self.log("train/auroc", auroc, prog_bar=True)

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
        try:
            self.val_auroc(preds, targets.long())
            auroc = self.val_auroc.compute()
        # If there is no positive instance
        except ValueError:
            auroc = 0
        # log val metrics
        self.log("val/prec", prec, prog_bar=True)
        self.log("val/rec", rec, prog_bar=True)
        self.log("val/f1", f1, prog_bar=True)
        self.log("val/auroc", auroc, prog_bar=True)

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
        try:
            self.test_auroc(preds, targets.long())
            auroc = self.test_auroc.compute()
        # If there is no positive instance
        except ValueError:
            auroc = 0
        # log test metrics
        self.log("test/prec", prec)
        self.log("test/rec", rec)
        self.log("test/f1", f1)
        self.log("test/auroc", auroc, prog_bar=True)

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
