import logging
from pathlib import Path
from typing import Dict, Union

import torch
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset

from .mixin import GraphDataMixin, LabelMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GraphDataset(Dataset, GraphDataMixin, LabelMixin):
    def __init__(
        self,
        csv_path: Union[str, Path],
        pdbid2node: Dict[str, torch.Tensor],
        pdbid2adj: Dict[str, coo_matrix],
    ):
        self.load_label(csv_path)
        self.load_pdb_data(csv_path, pdbid2node, pdbid2adj)

    def __getitem__(self, index):
        return (
            self.amino_acids_graph_list0[index],
            self.amino_acids_graph_list1[index],
            self.labels[index],
        )

    def __len__(self) -> int:
        return len(self.amino_acids_graph_list0)
