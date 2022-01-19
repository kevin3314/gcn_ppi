from pathlib import Path
from typing import Optional, Union

from torch_geometric.loader import DataLoader as GDataLoader

from src.datamodules.datasets.text_and_num_dataset import TextAndNumDataset
from src.datamodules.text_datamodule import TextDatasetModule


class TextAndNumModule(TextDatasetModule):
    def __init__(
        self,
        data_dir: Union[str, Path],
        feature_tsv_path: Union[str, Path],
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
        super().__init__(data_dir, pretrained_path, batch_size, num_workers, pin_memory, max_seq_len, **kwargs)
        self.feature_tsv_path = Path(feature_tsv_path)

    def setup(self, stage: Optional[str] = None):
        """Load data"""
        self.train_ds = TextAndNumDataset(
            self.data_dir / "train.csv",
            self.feature_tsv_path,
            self.tokenizer,
            max_seq_len=self.max_seq_len,
        )
        self.valid_ds = TextAndNumDataset(
            self.data_dir / "valid.csv",
            self.feature_tsv_path,
            self.tokenizer,
            max_seq_len=self.max_seq_len,
        )
        self.test_ds = TextAndNumDataset(
            self.data_dir / "test.csv",
            self.feature_tsv_path,
            self.tokenizer,
            max_seq_len=self.max_seq_len,
        )

    def train_dataloader(self):
        return GDataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return GDataLoader(
            dataset=self.valid_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return GDataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
