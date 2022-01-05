from pathlib import Path
from typing import Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.datamodules.datasets.text_dataset import TextDataset


class TextDatasetModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path],
        pretrained_path: str,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_seq_len: int = 256,
        **kwargs,
    ):
        """
        Args:
            data_dir (Union[str, Path]): Data dir to load.
            k (int, optional): The number of neighbor nodes. Defaults to 5.
            batch_size (int, optional): Batch sizes. Defaults to 32.
            num_workers (int, optional): The number of workers. Defaults to 0.
            pin_memory (bool, optional): Defaults to False.
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_seq_len = max_seq_len

        self.train_ds: Optional[Dataset] = None
        self.valid_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data"""
        self.train_ds = TextDataset(self.data_dir / "train.csv", self.tokenizer, max_seq_len=self.max_seq_len)
        self.valid_ds = TextDataset(self.data_dir / "valid.csv", self.tokenizer, max_seq_len=self.max_seq_len)
        self.test_ds = TextDataset(self.data_dir / "test.csv", self.tokenizer, max_seq_len=self.max_seq_len)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
