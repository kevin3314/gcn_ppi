from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .convert_text import build_edges_by_proteins, get_nodes_repr_for_texts
from .preprocess_on_graph import batch_graph, get_hop_distance, wl_node_coloring


class GraphClassificationDataset:
    def __init__(
        self,
        csv_path: Union[str, Path],
        k: Optional[int] = 5,
    ):
        self.load_data(csv_path, k)

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
        k: Optional[int] = 5,
    ) -> None:
        """Load data from data_path.

        Args:
            csv_path (Union[str, Path]): Csv path to load.
            k (Optional[int], optional): The number of neighbors. Defaults to 5.

        Returns:
            InputFeatures: Dataclass representing all input features.
        """
        df = pd.read_csv(csv_path)
        s_path = Path(csv_path).parent / f"s_{k}.pkl"
        nodes: np.ndarray = get_nodes_repr_for_texts(df["text"].values)
        node_ids = np.arange(nodes.shape[0])
        edges = build_edges_by_proteins(df["id"].values, df["protein0"].values, df["protein1"].values, s_path)
        labels = df["GOLD"]

        wl_dict: Dict[int, int] = wl_node_coloring(nodes, edges)
        batch_dict: Dict[int, List[int]] = batch_graph(nodes, s_path, k=k)
        hop_dict: Dict[int, Dict[int, int]] = get_hop_distance()

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
        self.role_ids = np.array(role_ids_list)
        self.position_ids = np.array(position_ids_list)
        self.hop_ids = np.array(hop_ids_list)
        self.labels = labels
