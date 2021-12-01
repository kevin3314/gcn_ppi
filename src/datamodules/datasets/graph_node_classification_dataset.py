from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix, load_npz
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

from .convert_text import build_edges_by_proteins, get_nodes_repr_for_texts
from .preprocess_on_graph import batch_graph, get_hop_distance, wl_node_coloring


class GraphNodeClassificationDataset(Dataset):
    def __init__(
        self,
        csv_path: Union[str, Path],
        split: str,
        k: Optional[int] = 5,
        pdb_processed_root: Optional[Union[Path, str]] = "/home/umakoshi/Documents/ppi/gcn_ppi/data/pdb_processed",
    ):
        self.load_data(csv_path, split, k, pdb_processed_root)

    def __len__(self):
        return len(self.raw_features)

    def __getitem__(self, index):
        raw_features = self.raw_features[index]  # (K, D)
        amino_acids_graph0 = self.amino_acids_graphs0[index]  # (A0)
        amino_acids_graph1 = self.amino_acids_graphs1[index]  # (A1)
        amino_acids_number0 = self.amino_acids_number_list0[index]
        amino_acids_number1 = self.amino_acids_number_list1[index]
        role_ids = self.role_ids[index]  # (K, 1)
        position_ids = self.position_ids[index]  # (K, 1)
        hop_ids = self.hop_ids[index]  # (K, 1)
        label = self.labels[index]  # (1)

        return (
            raw_features,
            amino_acids_graph0,
            amino_acids_graph1,
            amino_acids_number0,
            amino_acids_number1,
            role_ids,
            position_ids,
            hop_ids,
            label,
        )

    @staticmethod
    def get_pdb_nodes(
        pdb_ids0: List[str], pdb_ids1: List[str], pdb_root: Path
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        res0 = [
            np.load(pdb_root / f"{pdb_id}_ids.npy") if (pdb_root / f"{pdb_id}_ids.npy").exists() else np.zeros(0)
            for pdb_id in pdb_ids0
        ]
        res1 = [
            np.load(pdb_root / f"{pdb_id}_ids.npy") if (pdb_root / f"{pdb_id}_ids.npy").exists() else np.zeros(0)
            for pdb_id in pdb_ids1
        ]
        return res0, res1

    @staticmethod
    def get_adj_matrix(
        pdb_ids0: List[str], pdb_ids1: List[str], pdb_root: Path
    ) -> Tuple[List[coo_matrix], List[coo_matrix]]:
        res0 = [
            load_npz(pdb_root / f"{pdb_id}_adj.npz") if (pdb_root / f"{pdb_id}_adj.npz").exists() else np.zeros(0)
            for pdb_id in pdb_ids0
        ]
        res1 = [
            load_npz(pdb_root / f"{pdb_id}_adj.npz") if (pdb_root / f"{pdb_id}_adj.npz").exists() else np.zeros(0)
            for pdb_id in pdb_ids1
        ]
        return res0, res1

    def load_data(
        self,
        csv_path: Union[str, Path],
        split: str,
        k: Optional[int] = 5,
        pdb_processed_root: Optional[Union[Path, str]] = "/home/umakoshi/Documents/ppi/gcn_ppi/data/pdb_processed",
    ) -> None:
        """Load data from data_path.

        Args:
            csv_path (Union[str, Path]): Csv path to load.
            k (Optional[int], optional): The number of neighbors. Defaults to 5.

        Returns:
            InputFeatures: Dataclass representing all input features.
        """
        csv_path = Path(csv_path)
        pdb_processed_root = Path(pdb_processed_root)
        df = pd.read_csv(csv_path)
        s_path = csv_path.parent / f"s_{split}_{k}.pkl"
        pca_path: Optional[Path] = csv_path.parent / "pca.pkl"
        vocab_path: Optional[Path] = csv_path.parent / "vocab.pkl"

        # Text graph
        text_nodes: np.ndarray = get_nodes_repr_for_texts(
            df["text"].values,
            write_pca_path=pca_path if split == "train" else None,
            load_pca_path=pca_path if split != "train" else None,
            write_vocab_path=vocab_path if split == "train" else None,
            load_vocab_path=vocab_path if split != "train" else None,
        )
        # Protein graph.
        # Protein node is sum of two tensor representing each protein.
        # Each tensor is calculated by GNN on amino acid structure: node is embeddings of each amino acid and
        # adjacency matrix based on distance of amino acid.
        # amino_acids_list is (instances, 2, num_amino_acids).
        # amino_acids_adj_list is (instances, 2, num_nodes, num_nodes).
        # Dealing with length mismatch is done on aggregation.
        amino_acids_list0, amino_acids_list1 = self.get_pdb_nodes(
            df["PDB_ID0"].values, df["PDB_ID1"], pdb_processed_root
        )
        amino_acids_adj_list0, amino_acids_adj_list1 = self.get_adj_matrix(
            df["PDB_ID0"].values, df["PDB_ID1"], pdb_processed_root
        )
        amino_acids_number_list0: np.ndarray = np.cumsum([len(aa) for aa in amino_acids_list0])
        amino_acids_number_list1: np.ndarray = np.cumsum([len(aa) for aa in amino_acids_list1])

        amino_acids_edges0: List[torch.Tensor] = [from_scipy_sparse_matrix(adj)[0] for adj in amino_acids_adj_list0]
        amino_acids_edges1: List[torch.Tensor] = [from_scipy_sparse_matrix(adj)[0] for adj in amino_acids_adj_list1]
        amino_acids_graphs0: List[Data] = [
            Data(x=amino_acids, edge_index=edge_index)
            for amino_acids, edge_index in zip(amino_acids_list0, amino_acids_edges0)
        ]
        amino_acids_graphs1: List[Data] = [
            Data(x=amino_acids, edge_index=edge_index)
            for amino_acids, edge_index in zip(amino_acids_list1, amino_acids_edges1)
        ]

        text_node_ids = np.arange(text_nodes.shape[0])
        text_edges = build_edges_by_proteins(df["ID"].values, df["PROTEIN0"].values, df["PROTEIN1"].values, s_path)
        labels = df["GOLD"].astype(int)

        wl_dict: Dict[int, int] = wl_node_coloring(text_node_ids, text_edges)
        batch_dict: Dict[int, List[int]] = batch_graph(text_node_ids, s_path, k=k)
        hop_dict: Dict[int, Dict[int, int]] = get_hop_distance(text_node_ids, text_edges, batch_dict)

        # N: number of nodes, K: number of neighbors, D: dimension of feature
        raw_feature_list = []  # (N, K, D)
        role_ids_list = []  # (N, K)
        position_ids_list = []  # (N, K)
        hop_ids_list = []  # (N, K)
        for node_idx in text_node_ids:
            neighbors_list = batch_dict[node_idx]  # (K)

            raw_feature = [text_nodes[node_idx].tolist()]  # (K, D)
            role_ids = [wl_dict[node_idx]]  # (K)
            position_ids = range(len(neighbors_list) + 1)  # (K)
            hop_ids = [0]  # (K)
            for neighbor_idx, _ in neighbors_list:
                raw_feature.append(text_nodes[neighbor_idx].tolist())
                role_ids.append(wl_dict[neighbor_idx])
                if neighbor_idx in hop_dict[node_idx]:
                    hop_ids.append(hop_dict[node_idx][neighbor_idx])
                else:
                    hop_ids.append(99)
            raw_feature_list.append(raw_feature)
            role_ids_list.append(role_ids)
            position_ids_list.append(position_ids)
            hop_ids_list.append(hop_ids)

        self.raw_features = np.array(raw_feature_list)
        self.role_ids = np.array(role_ids_list).astype(np.int64)
        self.position_ids = np.array(position_ids_list).astype(np.int64)
        self.hop_ids = np.array(hop_ids_list).astype(np.int64)
        self.labels = labels.astype(np.float32)
        self.amino_acids_graphs0 = amino_acids_graphs0
        self.amino_acids_graphs1 = amino_acids_graphs1
        self.amino_acids_number_list0 = amino_acids_number_list0
        self.amino_acids_number_list1 = amino_acids_number_list1
