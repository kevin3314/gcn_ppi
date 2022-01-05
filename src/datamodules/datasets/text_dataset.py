import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextDataset(Dataset):
    def __init__(self, csv_path: Union[str, Path], tokenizer: PreTrainedTokenizerBase, max_seq_len: int = 256):
        self.load_data(csv_path, tokenizer, max_seq_len=max_seq_len)

    def load_data(self, csv_path: Union[str, Path], tokenizer: PreTrainedTokenizerBase, max_seq_len: int = 256):
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        texts = df["text"].values
        pad_token_id = tokenizer.pad_token_id

        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        exceeded_samples = 0

        # Tokenize
        for text in texts:
            inputs = tokenizer(text)
            input_ids = inputs["input_ids"]
            token_type_ids = inputs["token_type_ids"]
            attention_mask = inputs["attention_mask"]
            if len(input_ids) > max_seq_len:
                exceeded_samples += 1
                input_ids = input_ids[:max_seq_len]
                token_type_ids = token_type_ids[:max_seq_len]
                attention_mask = attention_mask[:max_seq_len]
            while len(input_ids) < max_seq_len:
                input_ids.append(pad_token_id)
                token_type_ids.append(0)
                attention_mask.append(0)
            self.input_ids.append(torch.Tensor(input_ids).long())
            self.token_type_ids.append(torch.Tensor(token_type_ids).long())
            self.attention_mask.append(torch.Tensor(attention_mask).long())

        logger.info(f"Found {exceeded_samples} samples exceeding max sequence length of {max_seq_len} in {csv_path}")
        labels = df["GOLD"].astype(np.float32)
        self.labels = torch.from_numpy(np.array(labels))

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.input_ids)
