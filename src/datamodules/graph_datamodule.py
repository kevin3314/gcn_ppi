from numbers import Number
from pathlib import Path
from typing import Optional, Union

from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader as GDataLoader

from src.datamodules.datasets.graph_dataset import GraphDataset
from src.datamodules.utils import construct_graph


class GraphDatasetModule(LightningDataModule):
    def __init__(
        self,
        train_csv_path: Union[str, Path],
        valid_csv_path: Union[str, Path],
        test_csv_path: Union[str, Path],
        pdb_path: Union[str, Path] = None,
        threshold: Number = 8,
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
        self.train_csv_path = Path(train_csv_path)
        self.valid_csv_path = Path(valid_csv_path)
        self.test_csv_path = Path(test_csv_path)
        self.threshold = threshold
        self.pdb_path = Path(pdb_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        """Load data"""
        pdbid2nodes, pdbid2adjs, _ = construct_graph(
            self.train_csv_path, self.valid_csv_path, self.test_csv_path, self.pdb_path, self.threshold
        )
        self.train_ds = GraphDataset(self.train_csv_path, pdbid2nodes, pdbid2adjs)
        self.valid_ds = GraphDataset(self.valid_csv_path, pdbid2nodes, pdbid2adjs)
        self.test_ds = GraphDataset(self.test_csv_path, pdbid2nodes, pdbid2adjs)

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
