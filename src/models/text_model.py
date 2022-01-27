import logging
from typing import Any

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import F1, Precision, Recall

from src.models.mixin import CommonMixin
from src.models.modules.text_module import TextModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextModule(CommonMixin, LightningModule):
    """This module works only with text modality."""

    def __init__(
        self,
        pretrained_path: str = "dmis-lab/biobert-v1.1",
        with_lstm: bool = False,
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

        self.train_prec = Precision()
        self.train_rec = Recall()
        self.train_f1 = F1()
        self.val_prec = Precision()
        self.val_rec = Recall()
        self.val_f1 = F1()
        self.test_prec = Precision()
        self.test_rec = Recall()
        self.test_f1 = F1()

        assert type(with_lstm) is bool
        self.model: TextModel = TextModel(pretrained_path, with_lstm)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(input_ids, token_type_ids, attention_mask)

    def step(self, batch: Any):
        input_ids, token_type_ids, attention_mask, labels = batch
        logits = self.forward(input_ids, token_type_ids, attention_mask)
        loss = self.criterion(logits, labels.float())
        preds = F.sigmoid(logits)
        # preds = torch.argmax(logits, dim=1)
        return loss, preds, labels
