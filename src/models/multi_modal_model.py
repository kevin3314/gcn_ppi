from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import F1, Precision, Recall
from transformers import AdamW, get_linear_schedule_with_warmup

from src.models.modules.graph_bert_layers import GraphBertConfig
from src.models.modules.mm_model import MultiModalModel


class MultiModalModule(LightningModule):
    def __init__(
        self,
        config: GraphBertConfig,
        amino_vocab_size: int,
        embedding_dim: int,
        num_gnn_layers: int,
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

        self.model: MultiModalModel = MultiModalModel(config, amino_vocab_size, embedding_dim, num_gnn_layers)
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
        amino_acids_graph_data0,
        amino_acids_graph_data1,
        role_ids: torch.Tensor,  # (batch_size, k)
        position_ids: torch.Tensor,  # (batch_size, k)
        hop_ids: torch.Tensor,  # (batch_size, k)
    ):
        return self.model(
            raw_features, amino_acids_graph_data0, amino_acids_graph_data1, role_ids, position_ids, hop_ids
        )

    def step(self, batch: Any):
        raw_features, amino_acids_graph_data0, amino_acids_graph_data1, role_ids, position_ids, hop_ids, labels = batch
        logits = self.forward(
            raw_features, amino_acids_graph_data0, amino_acids_graph_data1, role_ids, position_ids, hop_ids
        )
        loss = self.criterion(logits, labels.float())
        preds = (logits > 0.0).long()
        # preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        prec = self.train_prec(preds, targets.long())
        rec = self.train_rec(preds, targets.long())
        f1 = self.train_f1(preds, targets.long())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
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
        prec = self.train_prec(preds, targets.long())
        rec = self.train_rec(preds, targets.long())
        f1 = self.train_f1(preds, targets.long())
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
        prec = self.train_prec(preds, targets.long())
        rec = self.train_rec(preds, targets.long())
        f1 = self.train_f1(preds, targets.long())

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/prec", prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/rec", rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

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
