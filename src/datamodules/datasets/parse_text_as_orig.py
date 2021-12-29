from pathlib import Path
from typing import Union

import numpy as np
import scipy.sparse as sp
import torch
from numpy.linalg import inv

from .preprocess_on_graph import batch_graph, get_hop_distance, wl_node_coloring


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def adj_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def prosess_text_as_orig(
    idx_features_path: Union[str, Path], link_path: Union[str, Path], compute_s: bool = True, c: float = 0.15
):
    """Process node features and link following original implementation.

    Args:
        idx_features_path (Union[str, Path]): Text file where each line is feature.
        link_path (Union[str, Path]): Text file where each line is edge.
    """
    idx_features_labels = np.genfromtxt(idx_features_path, dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

    one_hot_labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    index_id_map = {i: j for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(link_path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(one_hot_labels.shape[0], one_hot_labels.shape[0]),
        dtype=np.float32,
    )

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    eigen_adj = None
    if compute_s:
        eigen_adj = c * inv((sp.eye(adj.shape[0]) - (1 - c) * adj_normalize(adj)).toarray())

    norm_adj = adj_normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(one_hot_labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(norm_adj)

    wl_dict = wl_node_coloring(idx, edges_unordered)
    batch_dict = batch_graph(index_id_map, S=eigen_adj)
    hop_dict = get_hop_distance(idx, edges_unordered, batch_dict)

    raw_feature_list = []
    role_ids_list = []
    position_ids_list = []
    hop_ids_list = []
    for node in idx:
        node_index = idx_map[node]
        neighbors_list = batch_dict[node]

        raw_feature = [features[node_index].tolist()]
        role_ids = [wl_dict[node]]
        position_ids = range(len(neighbors_list) + 1)
        hop_ids = [0]
        for neighbor, intimacy_score in neighbors_list:
            neighbor_index = idx_map[neighbor]
            raw_feature.append(features[neighbor_index].tolist())
            role_ids.append(wl_dict[neighbor])
            if neighbor in hop_dict[node]:
                hop_ids.append(hop_dict[node][neighbor])
            else:
                hop_ids.append(99)
        raw_feature_list.append(raw_feature)
        role_ids_list.append(role_ids)
        position_ids_list.append(position_ids)
        hop_ids_list.append(hop_ids)
    raw_embeddings = torch.FloatTensor(raw_feature_list)
    wl_embedding = torch.LongTensor(role_ids_list)
    hop_embeddings = torch.LongTensor(hop_ids_list)
    int_embeddings = torch.LongTensor(position_ids_list)

    return {
        "X": features,
        "A": adj,
        "S": eigen_adj,
        "index_id_map": index_id_map,
        "edges": edges_unordered,
        "raw_embeddings": raw_embeddings,
        "wl_embeddings": wl_embedding,
        "hop_embeddings": hop_embeddings,
        "int_embeddings": int_embeddings,
        "y": labels,
        "idx": idx,
    }
