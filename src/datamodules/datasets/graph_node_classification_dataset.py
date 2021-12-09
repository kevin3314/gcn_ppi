import logging
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


NULL_EMBEDDING = torch.ones(1)
NULL_ADJ = coo_matrix((0, 0))


class NeighborData(Data):
    def __init__(self, edge_indices: Optional[List[torch.Tensor]] = None, xs: Optional[List[torch.Tensor]] = None):
        super().__init__()
        # Null check is needed because instantination with
        # none arguments happens in complex inherence process.
        if xs is not None:
            for i, x in enumerate(xs):
                setattr(self, f"x_{i}", x)
        if edge_indices is not None:
            for i, edge in enumerate(edge_indices):
                setattr(self, f"edge_index_{i}", edge)

    def __inc__(self, key, value, *args, **kwargs):
        if "edge_index" in key:
            index = key.lstrip("edge_index_")
            x = getattr(self, f"x_{index}")
            return x.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphNodeClassificationDataset(Dataset):
    def __init__(
        self,
        csv_path: Union[str, Path],
        split: str,
        k: Optional[int] = 5,
        pdb_processed_root: Optional[Union[Path, str]] = "data/pdb_processed",
    ):
        self.load_data(csv_path, split, k, pdb_processed_root)

    def __len__(self):
        return len(self.raw_features)

    def __getitem__(self, index):
        raw_features = self.raw_features[index]  # (K, D)
        amino_acids_graph_data0 = self.amino_acids_graph_data0[index]  # (~K)
        amino_acids_graph_data1 = self.amino_acids_graph_data1[index]  # (~K)
        role_ids = self.role_ids[index]  # (K, 1)
        position_ids = self.position_ids[index]  # (K, 1)
        hop_ids = self.hop_ids[index]  # (K, 1)
        label = self.labels[index]  # (1)

        return (
            raw_features,
            amino_acids_graph_data0,
            amino_acids_graph_data1,
            role_ids,
            position_ids,
            hop_ids,
            label,
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

    def load_data(
        self,
        csv_path: Union[str, Path],
        split: str,
        k: Optional[int] = 5,
        pdb_processed_root: Optional[Union[Path, str]] = "data/pdb_processed",
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
        amino_acids_edges0: List[torch.Tensor] = [from_scipy_sparse_matrix(adj)[0] for adj in amino_acids_adj_list0]
        amino_acids_edges1: List[torch.Tensor] = [from_scipy_sparse_matrix(adj)[0] for adj in amino_acids_adj_list1]

        text_node_ids = np.arange(text_nodes.shape[0])
        # To handle proteins contained by each instance, use another instance's information
        # which share common text.
        text_edges: np.ndarray = build_edges_by_proteins(
            df["ID"].values,
            df["PROTEIN0"].values,
            df["PROTEIN1"].values,
            s_path,
        )
        labels = df["GOLD"].astype(int)

        wl_dict: Dict[int, int] = wl_node_coloring(text_node_ids, text_edges)
        batch_dict: Dict[int, List[int]] = batch_graph(text_node_ids, s_path, k=k)
        hop_dict: Dict[int, Dict[int, int]] = get_hop_distance(text_node_ids, text_edges, batch_dict)

        # N: number of nodes, K: number of neighbors, D: dimension of feature
        raw_feature_list = []  # (N, K, D)
        amino_acids_nodes_list0: List[List[torch.Tensor]] = []  # (N, K)
        amino_acids_nodes_list1: List[List[torch.Tensor]] = []  # (N, K)
        amino_acids_edges_list0: List[List[torch.Tensor]] = []  # (N, K, 2, num_edges)
        amino_acids_edges_list1: List[List[torch.Tensor]] = []  # (N, K, 2, num_edges)
        role_ids_list = []  # (N, K)
        position_ids_list = []  # (N, K)
        hop_ids_list = []  # (N, K)
        for node_idx in text_node_ids:
            neighbors_list = batch_dict[node_idx]  # (K)

            raw_feature = [text_nodes[node_idx].tolist()]  # (K, D)
            amino_acids_nodes0 = [amino_acids_list0[node_idx]]  # (K, A0)
            amino_acids_nodes1 = [amino_acids_list1[node_idx]]  # (K, A1)
            _amino_acids_edges0 = [amino_acids_edges0[node_idx]]  # (K, 2, num_edges)
            _amino_acids_edges1 = [amino_acids_edges1[node_idx]]  # (K, 2, num_edges)
            role_ids = [wl_dict[node_idx]]  # (K)
            position_ids = range(len(neighbors_list) + 1)  # (K)
            hop_ids = [0]  # (K)
            for neighbor_idx, _ in neighbors_list:
                raw_feature.append(text_nodes[neighbor_idx].tolist())
                amino_acids_nodes0.append(amino_acids_list0[neighbor_idx])
                amino_acids_nodes1.append(amino_acids_list1[neighbor_idx])
                _amino_acids_edges0.append(amino_acids_edges0[neighbor_idx])
                _amino_acids_edges1.append(amino_acids_edges1[neighbor_idx])
                role_ids.append(wl_dict[neighbor_idx])
                if neighbor_idx in hop_dict[node_idx]:
                    hop_ids.append(hop_dict[node_idx][neighbor_idx])
                else:
                    hop_ids.append(99)
            raw_feature_list.append(raw_feature)
            amino_acids_nodes_list0.append(amino_acids_nodes0)
            amino_acids_nodes_list1.append(amino_acids_nodes1)
            amino_acids_edges_list0.append(_amino_acids_edges0)
            amino_acids_edges_list1.append(_amino_acids_edges1)
            role_ids_list.append(role_ids)
            position_ids_list.append(position_ids)
            hop_ids_list.append(hop_ids)

        self.raw_features = torch.from_numpy(np.array(raw_feature_list))
        self.role_ids = torch.from_numpy(np.array(role_ids_list).astype(np.int64))
        self.position_ids = torch.from_numpy(np.array(position_ids_list).astype(np.int64))
        self.hop_ids = torch.from_numpy(np.array(hop_ids_list).astype(np.int64))
        self.labels = torch.from_numpy(np.array(labels.astype(np.float32)))

        assert all(x is not None for x in amino_acids_nodes_list0)
        assert all(x is not None for x in amino_acids_nodes_list1)
        assert all(x is not None for x in amino_acids_edges_list0)
        assert all(x is not None for x in amino_acids_edges_list1)
        self.amino_acids_graph_data0: List[NeighborData] = [
            NeighborData(_amino_acids_edges0, _amino_acids_nodes0)
            for (_amino_acids_nodes0, _amino_acids_edges0) in zip(amino_acids_nodes_list0, amino_acids_edges_list0)
        ]
        self.amino_acids_graph_data1: List[NeighborData] = [
            NeighborData(_amino_acids_edges1, _amino_acids_nodes1)
            for (_amino_acids_nodes1, _amino_acids_edges1) in zip(amino_acids_nodes_list1, amino_acids_edges_list1)
        ]
        logger.info(f"Successfully load data: {split}")
