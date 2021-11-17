import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np
import scipy as sp
import sklearn


def get_nodes_repr_for_texts(
    texts: List[str],
    node_dim: Optional[int] = 1000,
) -> np.ndarray:
    """Get dense node representation for texts.

    Args:
        texts (List[str]): Texts to proess.
        node_dim (int): Dimension of node.
        data_dir (Union[str, Path]): Directory to save S.

    Returns:
        np.ndarray: Numpy array of node representation with shape of
                    (nodes, node_dim)
    """
    _, bag_of_words = convert_text_to_one_hot_vector(texts)
    result: np.ndarray = sklearn.decomposition.PCA(bag_of_words)
    # Retrieve top-node_dim components
    result = result[:, :node_dim]
    return result


def convert_text_to_one_hot_vector(
    texts: List[str],
) -> Tuple[Dict[str, int], np.ndarray]:
    """Convert text to one hot vector with vocabulary.add()

    Args:
        text (List[str]): Text to process.
        data_dir (Union[str, Path]): Directory to save S.

    Returns:
        Tuple[str, List[np.ndarray]]: Tuple of vocabulary and processed ndarray.
    """
    vocab: Dict[str, int] = {}
    count: int = 0
    # At first, determine the size of vocabulary.
    vocab_size = len(set(word for text in texts for word in text.strip().split()))

    # Bag of words for each sample
    bag_of_words: List[np.ndarray] = [np.zeros(vocab_size) for _ in range(len(texts))]
    for text, bag_of_word in zip(texts, bag_of_words):
        for word in text.strip().split():
            if word not in vocab:
                vocab[word] = count
                count += 1
            bag_of_words[vocab[word]] = 1

    return vocab, np.array(bag_of_words)


def build_edges_by_proteins(
    ids: Sequence[str],
    protein0s: Sequence[int],
    pritein1s: Sequence[int],
    s_path: Optional[Union[str, Path]] = None,
    alpha: Optional[float] = None,
) -> np.ndarray:
    """Build edges based on whether instance shares same protein.

    MEMO:
        Convert protein name to index should boost performance.

    Args:
        ids (Sequence[int]): Ids of instances.
        protein0s (Sequence[int]): Protein ids of instance 0.
        pritein1s (Sequence[int]): Protein ids of instance 1.
        s_path (Optional[Union[str, Path]]): Path to save S.
        alpha: (Optional[float]): Coefficient to calculate S.

    Returns:
        np.ndarray: Numpy array of edges.
    """
    # Constract all proteins for each ids (instance with same id share under text)
    # Then substruct targeted protein's count
    id2all_protein_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _id, protein0, protein1 in zip(ids, protein0s, pritein1s):
        id2all_protein_counts[_id][protein0] += 1
        id2all_protein_counts[_id][protein1] += 1
    contain_protein_counts = []
    for _id, protein0, protein1 in zip(ids, protein0s, pritein1s):
        contain_protein_counts.append(id2all_protein_counts[_id])
        contain_protein_counts[-1][protein0] -= 1
        contain_protein_counts[-1][protein1] -= 1
    contain_protein_sets: List[Set[str]] = [
        set(key for key, value in contain_protein_count if value > 0)
        for contain_protein_count in contain_protein_counts
    ]
    adj = np.zeros_like((len(ids), len(ids)))

    # Build edges based on contain_protein_sets
    for i, contain_protein_set0 in enumerate(contain_protein_sets):
        for _j, contain_protein_set1 in enumerate(contain_protein_sets[i + 1 :]):
            if len(contain_protein_set0 & contain_protein_set1) > 0:
                j = i + _j + 1
                adj[i][j] = 1
                adj[j][j] = 1
    D = nx.DiGraph(adj)
    edges = np.array([(u, v) for (u, v) in D.edges()])

    if s_path is not None and os.path.exists(s_path):

        def adj_normalize(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -0.5).flatten()
            r_inv[np.isinf(r_inv)] = 0.0
            r_mat_inv = sp.diags(r_inv)
            mx = r_mat_inv.dot(mx).dot(r_mat_inv)
            return mx

        assert alpha is not None
        # Calculate S
        adj = sp.coo_matrix(adj)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        eigen_adj = alpha * np.linalg.inv((sp.eye(adj.shape[0]) - (1 - alpha) * adj_normalize(adj)).toarray())
        with open(s_path, "w") as f:
            pickle.dump(eigen_adj, f)

    return edges
