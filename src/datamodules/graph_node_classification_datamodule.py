from pathlib import Path
from typing import Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.datasets.graph_node_classification_dataset import (
    GraphNodeClassificationDataset,
)


class GraphNodeClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path],
        k: int = 5,
        pdb_processed_root: Optional[Union[Path, str]] = "/home/umakoshi/Documents/ppi/gcn_ppi/data/pdb_processed",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs
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
        self.pdb_processed_root = Path(pdb_processed_root)
        self.k = k
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_ds: Optional[Dataset] = None
        self.valid_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data"""
        self.train_ds = GraphNodeClassificationDataset(
            self.data_dir / "train.csv", "train", self.k, self.pdb_processed_root
        )
        self.valid_ds = GraphNodeClassificationDataset(
            self.data_dir / "valid.csv", "valid", self.k, self.pdb_processed_root
        )
        self.test_ds = GraphNodeClassificationDataset(
            self.data_dir / "test.csv", "test", self.k, self.pdb_processed_root
        )

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
