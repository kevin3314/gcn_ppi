from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class TextDataset(Dataset):
    def __init__(self, csv_path: Union[str, Path], tokenizer: PreTrainedTokenizerBase):
        self.load_data(csv_path, tokenizer)

    def load_data(self, csv_path: Union[str, Path], tokenizer: PreTrainedTokenizerBase):
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        texts = df["text"].values
        # Tokenize
        self.inputs = [tokenizer(text) for text in texts]
        labels = df["GOLD"].astype(np.float32)
        self.labels = torch.from_numpy(np.array(labels))

    def __getitem__(self, index):
        return self.input_ids[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.input_ids)
