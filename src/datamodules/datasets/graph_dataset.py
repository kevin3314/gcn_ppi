from pathlib import Path
from typing import Union

from torch.utils.data import Dataset

from .mixin import GraphDataMixin, LabelMixin


class GraphDataset(Dataset, GraphDataMixin, LabelMixin):
    def __init__(
        self,
        csv_path: Union[str, Path],
        pdb_processed_root: Union[str, Path],
    ):
        self.load_label(csv_path)
        self.load_pdb_data(csv_path, pdb_processed_root)

    def __getitem__(self, index):
        return (
            self.amino_acids_graph_list0[index],
            self.amino_acids_graph_list1[index],
            self.labels[index],
        )

    def __len__(self) -> int:
        return len(self.amino_acids_graph_list0)