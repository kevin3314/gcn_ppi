from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .convert_text import build_edges_by_proteins, get_nodes_repr_for_texts
from .preprocess_on_graph import batch_graph, get_hop_distance, wl_node_coloring


class GraphNodeClassificationDataset(Dataset):
    def __init__(
        self,
        csv_path: Union[str, Path],
        split: str,
        k: Optional[int] = 5,
    ):
        self.load_data(csv_path, split, k)

    def __len__(self):
        return len(self.raw_features)

    def __getitem__(self, index):
        raw_features = self.raw_features[index]  # (K, D)
        role_ids = self.role_ids[index]  # (K, 1)
        position_ids = self.position_ids[index]  # (K, 1)
        hop_ids = self.hop_ids[index]  # (K, 1)
        label = self.labels[index]  # (1)
        return raw_features, role_ids, position_ids, hop_ids, label

    def load_data(
        self,
        csv_path: Union[str, Path],
        split: str,
        k: Optional[int] = 5,
    ) -> None:
        """Load data from data_path.

        Args:
            csv_path (Union[str, Path]): Csv path to load.
            k (Optional[int], optional): The number of neighbors. Defaults to 5.

        Returns:
            InputFeatures: Dataclass representing all input features.
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        s_path = csv_path.parent / f"s_{split}_{k}.pkl"
        pca_path: Optional[Path] = csv_path.parent / "pca.pkl"
        vocab_path: Optional[Path] = csv_path.parent / "vocab.pkl"

        nodes: np.ndarray = get_nodes_repr_for_texts(
            df["text"].values,
            write_pca_path=pca_path if split == "train" else None,
            load_pca_path=pca_path if split != "train" else None,
            write_vocab_path=vocab_path if split == "train" else None,
            load_vocab_path=vocab_path if split != "train" else None,
        )
        node_ids = np.arange(nodes.shape[0])
        edges = build_edges_by_proteins(df["ID"].values, df["PROTEIN0"].values, df["PROTEIN1"].values, s_path)
        labels = df["GOLD"].astype(int)

        wl_dict: Dict[int, int] = wl_node_coloring(node_ids, edges)
        batch_dict: Dict[int, List[int]] = batch_graph(node_ids, s_path, k=k)
        hop_dict: Dict[int, Dict[int, int]] = get_hop_distance(node_ids, edges, batch_dict)

        # N: number of nodes, K: number of neighbors, D: dimension of feature
        raw_feature_list = []  # (N, K, D)
        role_ids_list = []  # (N, K)
        position_ids_list = []  # (N, K)
        hop_ids_list = []  # (N, K)
        for node_idx in node_ids:
            neighbors_list = batch_dict[node_idx]  # (K)

            raw_feature = [nodes[node_idx].tolist()]  # (K, D)
            role_ids = [wl_dict[node_idx]]  # (K)
            position_ids = range(len(neighbors_list) + 1)  # (K)
            hop_ids = [0]  # (K)
            for neighbor_idx, _ in neighbors_list:
                raw_feature.append(nodes[neighbor_idx].tolist())
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
