"""From https://github.com/jwzhanggy/Graph-Bert/blob/e3e5fc57b2cb27f86b38bd87982be1d82303df3d/code/MethodHopDistance.py
"""

import hashlib
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
from numpy import ndarray


def wl_node_coloring(
    ids: Sequence[int],
    edges: Sequence[Sequence[int]],
    max_iter: Optional[int] = 2,
) -> Dict[int, int]:
    """WL node coloring.

    Args:
        ids (Sequence[int]): Node ids list.
        edges (Sequence[Sequence[int]]): Edge List.
        max_iter (Optional[int], optional): Max iterations of algorithm. Defaults to 2.

    Returns:
        Dict[int, int]: Mapping from node id to color id.
    """
    node_color_dict = {}
    node_neighbor_dict = {}

    for node in ids:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for (u1, u2) in edges:
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    # NOTE: When max_iter is set to 2, it halt after the first iteration.
    # This behavior is identical to the original implementation.
    iteration_count = 1
    while True:
        new_color_dict = {}
        for node in ids:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v + 1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict:
            return new_color_dict
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
        if iteration_count == max_iter:
            return node_color_dict


def batch_graph(
    ids: Sequence[int], s_path: Union[str, Path], k: Optional[int] = 5
) -> Dict[int[List[Tuple[int, ndarray]]]]:
    """Batch graph by top-k sampling based on S matrix.

    Args:
        ids (Sequence[int]): Node ids list.
        s_path (Union[str, Path]): Path to load S matrix.
        k (Optional[int]): Top-k sampling. Defaults to 5.

    Returns:
        Dict[int[List[Tuple[int, ndarray]]]]: Mapping from node id to list of tuple
                                             (neighbor_id, s_value)
    """
    with open(s_path, "rb") as f:
        S = pickle.load(f)
    user_top_k_neighbor_intimacy_dict = {}
    for node_index in ids:
        s = S[node_index]
        s[node_index] = -1000.0
        top_k_neighbor_index = s.argsort()[-k:][::-1]
        user_top_k_neighbor_intimacy_dict[node_index] = []
        for neighbor_index in top_k_neighbor_index:
            user_top_k_neighbor_intimacy_dict[node_index].append((neighbor_index, s[neighbor_index]))
    return user_top_k_neighbor_intimacy_dict


def get_hop_distance(
    ids: Sequence[int],
    edges: Sequence[Sequence[int]],
    batch_dict: Dict[int[List[Tuple[int, ndarray]]]],
) -> Dict[int[Dict[int, int]]]:
    """Get hop distance given S.

    Args:
        ids (Sequence[int]): Node ids list.
        edges (Sequence[Sequence[int]]): Edge List.
        batch_dict_path (Union[str, Path]): Path to load batch dict contains maping
                                            from node ids to neighbor node ids.

    Returns:
        Dict[int, Dict[int, int]]: Map from node id to neigbors to hop distance.
    """
    G = nx.Graph()
    G.add_nodes_from(ids)
    G.add_edges_from(edges)
    hop_dict = defaultdict(dict)
    for node in batch_dict:
        for neighbor, _ in batch_dict[node]:
            try:
                hop = nx.shortest_path_length(G, source=node, target=neighbor)
            except Exception:
                hop = 99
            hop_dict[node][neighbor] = hop
    return hop_dict
