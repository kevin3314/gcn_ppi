from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix, load_npz
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .text_dataset import TextDataset

NULL_EMBEDDING = torch.ones(1)
NULL_ADJ = coo_matrix((0, 0))


class TextAndGraphDataset(TextDataset):
    def __init__(
        self,
        csv_path: Union[str, Path],
        pdb_processed_root: Union[str, Path],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 256,
    ):
        super().__init__(csv_path, tokenizer, max_seq_len)
        df = pd.read_csv(Path(csv_path))
        self.load_pdb_data(df, pdb_processed_root)

    def __getitem__(self, index):
        return (
            self.input_ids[index],
            self.token_type_ids[index],
            self.attention_mask[index],
            self.amino_acids_graph_list0[index],
            self.amino_acids_graph_list1[index],
            self.labels[index],
        )

    @staticmethod
    def get_pdb_nodes(
        pdb_ids0: List[str], pdb_ids1: List[str], pdb_root: Path
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        res0 = [
            torch.from_numpy(np.load(pdb_root / f"{pdb_id}_ids.npy"))
            if (pdb_root / f"{pdb_id}_ids.npy").exists()
            else NULL_EMBEDDING
            for pdb_id in pdb_ids0
        ]
        res1 = [
            torch.from_numpy(np.load(pdb_root / f"{pdb_id}_ids.npy"))
            if (pdb_root / f"{pdb_id}_ids.npy").exists()
            else NULL_EMBEDDING
            for pdb_id in pdb_ids1
        ]
        return res0, res1

    @staticmethod
    def get_adj_matrix(
        pdb_ids0: List[str], pdb_ids1: List[str], pdb_root: Path
    ) -> Tuple[List[coo_matrix], List[coo_matrix]]:
        res0 = [
            load_npz(pdb_root / f"{pdb_id}_adj.npz") if (pdb_root / f"{pdb_id}_adj.npz").exists() else NULL_ADJ
            for pdb_id in pdb_ids0
        ]
        res1 = [
            load_npz(pdb_root / f"{pdb_id}_adj.npz") if (pdb_root / f"{pdb_id}_adj.npz").exists() else NULL_ADJ
            for pdb_id in pdb_ids1
        ]
        return res0, res1

    def load_pdb_data(self, df: pd.DataFrame, pdb_processed_root: Union[str, Path]):
        amino_acids_list0, amino_acids_list1 = self.get_pdb_nodes(
            df["PDB_ID0"].values, df["PDB_ID1"], pdb_processed_root
        )
        amino_acids_adj_list0, amino_acids_adj_list1 = self.get_adj_matrix(
            df["PDB_ID0"].values, df["PDB_ID1"], pdb_processed_root
        )
        amino_acids_edges0: List[torch.Tensor] = [from_scipy_sparse_matrix(adj)[0] for adj in amino_acids_adj_list0]
        amino_acids_edges1: List[torch.Tensor] = [from_scipy_sparse_matrix(adj)[0] for adj in amino_acids_adj_list1]

        self.amino_acids_graph_list0 = [
            Data(amino_acids_node, amino_acids_edge)
            for amino_acids_node, amino_acids_edge in zip(amino_acids_list0, amino_acids_edges0)
        ]
        self.amino_acids_graph_list1 = [
            Data(amino_acids_node, amino_acids_edge)
            for amino_acids_node, amino_acids_edge in zip(amino_acids_list1, amino_acids_edges1)
        ]
