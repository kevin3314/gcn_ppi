from pathlib import Path
from typing import Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as GDataLoader

from src.datamodules.datasets.parse_text_as_orig import prosess_text_as_orig
from src.datamodules.datasets.raw_graph_node_classification_dataset import (
    RawGraphNodeClassificationDataset,
)


class GraphNodeClassificationByOrigFormatDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path],
        train_split: int = 800,
        valid_split: int = 200,
        k: int = 5,
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

        self.data_dir = Path(data_dir)
        self.train_split = train_split
        self.valid_split = valid_split
        self.k = k
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_ds: Optional[Dataset] = None
        self.valid_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data"""
        processed = prosess_text_as_orig(self.data_dir / "node", self.data_dir / "link")
        train_raw_embeddings = processed["raw_embeddings"][: self.train_split]
        train_wl_embeddings = processed["wl_embeddings"][: self.train_split]
        train_hop_embeddings = processed["hop_embeddings"][: self.train_split]
        train_int_embeddings = processed["int_embeddings"][: self.train_split]
        train_label = processed["y"][: self.train_split]

        valid_raw_embeddings = processed["raw_embeddings"][self.train_split : self.train_split + self.valid_split]
        valid_wl_embeddings = processed["wl_embeddings"][self.train_split : self.train_split + self.valid_split]
        valid_hop_embeddings = processed["hop_embeddings"][self.train_split : self.train_split + self.valid_split]
        valid_int_embeddings = processed["int_embeddings"][self.train_split : self.train_split + self.valid_split]
        valid_label = processed["y"][: self.valid_split]

        test_raw_embeddings = processed["raw_embeddings"][self.train_split + self.valid_split :]
        test_wl_embeddings = processed["wl_embeddings"][self.train_split + self.valid_split :]
        test_hop_embeddings = processed["hop_embeddings"][self.train_split + self.valid_split :]
        test_int_embeddings = processed["int_embeddings"][self.train_split + self.valid_split :]
        test_label = processed["y"][self.train_split + self.valid_split :]

        self.train_ds = RawGraphNodeClassificationDataset(
            train_raw_embeddings, train_wl_embeddings, train_hop_embeddings, train_int_embeddings, train_label
        )
        self.valid_ds = RawGraphNodeClassificationDataset(
            valid_raw_embeddings, valid_wl_embeddings, valid_hop_embeddings, valid_int_embeddings, valid_label
        )
        self.test_ds = RawGraphNodeClassificationDataset(
            test_raw_embeddings, test_wl_embeddings, test_hop_embeddings, test_int_embeddings, test_label
        )

    def train_dataloader(self):
        return GDataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # One instance includes k neighboor nodes and target node (= k+1 nodes).
            follow_batch=[f"x_{i}" for i in range(self.k + 1)],
            shuffle=False,
        )

    def val_dataloader(self):
        return GDataLoader(
            dataset=self.valid_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # One instance includes k neighboor nodes and target node (= k+1 nodes).
            follow_batch=[f"x_{i}" for i in range(self.k + 1)],
            shuffle=False,
        )

    def test_dataloader(self):
        return GDataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # One instance includes k neighboor nodes and target node (= k+1 nodes).
            follow_batch=[f"x_{i}" for i in range(self.k + 1)],
            shuffle=False,
        )
