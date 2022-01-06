import logging
from pathlib import Path
from typing import Union

from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .mixin import LabelMixin, TextMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextDataset(Dataset, TextMixin, LabelMixin):
    def __init__(self, csv_path: Union[str, Path], tokenizer: PreTrainedTokenizerBase, max_seq_len: int = 256):
        self.load_text(csv_path, tokenizer, max_seq_len=max_seq_len)
        self.load_label(csv_path)

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.input_ids)
