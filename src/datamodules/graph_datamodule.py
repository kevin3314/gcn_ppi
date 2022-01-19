from pathlib import Path
from typing import Optional, Union

from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader as GDataLoader

from src.datamodules.datasets.graph_dataset import GraphDataset


class GraphDatasetModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path],
        pdb_processed_root: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        """
        Args:
            data_dir (Union[str, Path]): Data dir to load.
            batch_size (int, optional): Batch sizes. Defaults to 32.
            num_workers (int, optional): The number of workers. Defaults to 0.
            pin_memory (bool, optional): Defaults to False.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.pdb_processed_root = Path(pdb_processed_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        """Load data"""
        self.train_ds = GraphDataset(self.data_dir / "train.csv", self.pdb_processed_root)
        self.valid_ds = GraphDataset(self.data_dir / "valid.csv", self.pdb_processed_root)
        self.test_ds = GraphDataset(self.data_dir / "test.csv", self.pdb_processed_root)

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
