from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.preprocess.preprocess_on_graph import batch_graph, get_hop_distance, wl_node_coloring

from .convert_text import build_edges_by_proteins, get_nodes_repr_for_texts


@dataclass
class TrainExample:
    raw_feature: np.ndarray
    role_ids: np.ndarray
    position_ids: np.ndarray
    hop_ids: np.ndarray


def load_data(
    csv_path: Union[str, Path],
    k: Optional[int] = 5,
) -> TrainExample:
    """Load data from data_path.

    Args:
        data_path (Union[str, Path]): [description]

    Returns:
        GraphExample: [description]
    """
    df = pd.read_csv(csv_path)
    s_path = Path(csv_path).parent / f"s_{k}.pkl"
    nodes: np.ndarray = get_nodes_repr_for_texts(df["text"].values)
    node_ids = np.arange(nodes.shape[0])
    edges = build_edges_by_proteins(df["id"].values, df["protein0"].values, df["protein1"].values, s_path)

    wl_dict: Dict[int, int] = wl_node_coloring(nodes, edges)
    batch_dict: Dict[int, List[int]] = batch_graph(nodes, s_path, k=k)
    hop_dict: Dict[int, Dict[int, int]] = get_hop_distance()

    raw_feature_list = []
    role_ids_list = []
    position_ids_list = []
    hop_ids_list = []
    for node_idx in node_ids:
        neighbors_list = batch_dict[node_idx]

        raw_feature = [nodes[node_idx].tolist()]
        role_ids = [wl_dict[node_idx]]
        position_ids = range(len(neighbors_list) + 1)
        hop_ids = [0]
        for neighbor_idx, intimacy_score in neighbors_list:
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

    raw_feature_list = np.array(raw_feature_list)
    role_ids_list = np.array(role_ids_list)
    position_ids_list = np.array(position_ids_list)
    hop_ids_list = np.array(hop_ids_list)

    return TrainExample(
        raw_feature=raw_feature_list,
        role_ids=role_ids_list,
        position_ids=position_ids_list,
        hop_ids=hop_ids_list,
    )
