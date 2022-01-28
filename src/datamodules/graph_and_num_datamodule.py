from pathlib import Path
from typing import Optional, Union

from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader as GDataLoader

from src.datamodules.datasets.graph_and_num_dataset import GraphAndNumDataset


class GraphAndNumModule(LightningDataModule):
    def __init__(
        self,
        train_csv_path: Union[str, Path],
        valid_csv_path: Union[str, Path],
        test_csv_path: Union[str, Path],
        feature_tsv_path: Union[str, Path],
        pdb_processed_root: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
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
        self.train_csv_path = Path(train_csv_path)
        self.valid_csv_path = Path(valid_csv_path)
        self.test_csv_path = Path(test_csv_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.feature_tsv_path = Path(feature_tsv_path)
        self.pdb_processed_root = Path(pdb_processed_root)

    def setup(self, stage: Optional[str] = None):
        """Load data"""
        self.train_ds = GraphAndNumDataset(self.train_csv_path, self.feature_tsv_path, self.pdb_processed_root)
        self.valid_ds = GraphAndNumDataset(self.valid_csv_path, self.feature_tsv_path, self.pdb_processed_root)
        self.test_ds = GraphAndNumDataset(self.test_csv_path, self.feature_tsv_path, self.pdb_processed_root)

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
