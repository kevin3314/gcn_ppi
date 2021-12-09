import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np
import scipy.sparse as sps
import sklearn.decomposition

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_nodes_repr_for_texts(
    texts: List[str],
    node_dim: Optional[int] = 1000,
    write_pca_path: Optional[Union[str, Path]] = None,
    load_pca_path: Optional[Union[str, Path]] = None,
    write_vocab_path: Optional[Union[str, Path]] = None,
    load_vocab_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Get dense node representation for texts.
    If you want to write or overwrite pca instance, set wrtie_pca_path.

    Args:
        texts (List[str]): Texts to proess.
        node_dim (int): Dimension of node.
        write_pca_path: (Optional[Union[str, Path]]): Path to save pca.
        load_pca_path: (Optional[Union[str, Path]]): Path to load pca.
        write_vocab_path: (Optional[Union[str, Path]]): Path to save vocabulary.
        load_vocab_path: (Optional[Union[str, Path]]): Path to load vocabulary.

    Returns:
        np.ndarray: Numpy array of node representation with shape of
                    (nodes, node_dim)
    """
    assert write_pca_path is not None or load_pca_path is not None
    _, bag_of_words = convert_text_to_one_hot_vector(
        texts,
        write_vocab_path,
        load_vocab_path,
    )
    if write_pca_path:
        pca: sklearn.decomposition.PCA = sklearn.decomposition.PCA()
        pca.fit(bag_of_words)
        with open(write_pca_path, "wb") as f:
            pickle.dump(pca, f)
    else:
        with open(load_pca_path, "rb") as f:
            pca: sklearn.decomposition.PCA = pickle.load(f)
    result: np.ndarray = pca.transform(bag_of_words)
    # Retrieve top-node_dim components
    result = result[:, :node_dim]
    return result


def convert_text_to_one_hot_vector(
    texts: List[str],
    write_vocab_path: Optional[Union[str, Path]] = None,
    load_vocab_path: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, int], np.ndarray]:
    """Convert text to one hot vector with vocabulary.add()

    Args:
        text (List[str]): Text to process.
        data_dir (Union[str, Path]): Directory to save S.

    Returns:
        Tuple[str, List[np.ndarray]]: Tuple of vocabulary and processed ndarray.
    """
    assert write_vocab_path is not None or load_vocab_path is not None
    if write_vocab_path is not None:
        words = set(word for text in texts for word in text.strip().split())
        vocab = {word: i for i, word in enumerate(words)}
        with open(write_vocab_path, "wb") as f:
            pickle.dump(vocab, f)
    else:
        with open(load_vocab_path, "rb") as f:
            vocab = pickle.load(f)

    vocab_size = len(vocab)
    unk_idx = vocab_size
    # Onehot-like representation for each sample
    bag_of_words: List[np.ndarray] = [np.zeros(vocab_size + 1) for _ in range(len(texts))]
    for text, bag_of_word in zip(texts, bag_of_words):
        for word in text.strip().split():
            bag_of_word[vocab.get(word, unk_idx)] = 1

    return vocab, np.array(bag_of_words)


def build_edges_by_proteins(
    ids: Sequence[str],
    protein0s: Sequence[int],
    protein1s: Sequence[int],
    ref0s: Sequence[str],
    ref1s: Sequence[str],
    s_path: Optional[Union[str, Path]] = None,
    alpha: float = 0.15,
) -> np.ndarray:
    """Build edges based on whether instance shares same protein.

    MEMO:
        Convert protein name to index should boost performance.

    Args:
        ids (Sequence[int]): Ids of instances.
        protein0s (Sequence[int]): Protein ids of instance 0.
        protein1s (Sequence[int]): Protein ids of instance 1.
        ref0s (Sequence[str]): Reference text of each protein0.
        ref1s (Sequence[str]): Reference text of each protein1.
        s_path (Optional[Union[str, Path]]): Path to save S.
        alpha: (Optional[float]): Coefficient to calculate S.

    Returns:
        np.ndarray: Numpy array of edges. The number of nodes is len(ids).
    """
    # Constract all proteins for each ids (instance with same id share under text)
    # Then substruct targeted protein's count
    # (ids, 2)
    id2all_protein_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # To check whether target protein has already been seen.
    # Some protein is missing, so remember reference to protein correspondence.
    id2read_references: Dict[str, Dict[str, str]] = defaultdict(dict)
    for _id, protein0, protein1, ref0, ref1 in zip(ids, protein0s, protein1s, ref0s, ref1s):
        if ref0 not in id2read_references[_id]:
            id2read_references[_id][ref0] = protein0
            id2all_protein_counts[_id][protein0] += 1
        if ref1 not in id2read_references[_id]:
            id2read_references[_id][ref1] = protein1
            id2all_protein_counts[_id][protein1] += 1
    # (num_instances)
    contain_protein_counts = []
    for _id, protein0, protein1, ref0, ref1 in zip(ids, protein0s, protein1s, ref0s, ref1s):
        contain_protein_counts.append(dict(id2all_protein_counts[_id]))
        try:
            contain_protein_counts[-1][protein0] -= 1
        except KeyError:
            logger.debug("Found missing protein name. Try to recover by reference.")
            logger.debug("_id: %s, ref0: %s, protein0: %s", _id, ref0, protein0)
            logger.debug("id2read_references[_id]: %s", id2read_references[_id])
            contain_protein_counts[-1][id2read_references[_id][ref0]] -= 1
        try:
            contain_protein_counts[-1][protein1] -= 1
        except KeyError:
            logger.debug("Found missing protein name. Try to recover by reference.")
            logger.debug("_id: %s, ref1: %s, protein1: %s", _id, ref1, protein1)
            logger.debug("id2read_references[_id]: %s", id2read_references[_id])
            contain_protein_counts[-1][id2read_references[_id][ref1]] -= 1
    contain_protein_sets: List[Set[str]] = [
        set(key for key, value in contain_protein_count.items() if value > 0)
        for contain_protein_count in contain_protein_counts
    ]
    adj = np.zeros((len(ids), len(ids)))

    # Build edges based on contain_protein_sets
    for i, contain_protein_set0 in enumerate(contain_protein_sets):
        for _j, contain_protein_set1 in enumerate(contain_protein_sets[i + 1 :]):
            if len(contain_protein_set0 & contain_protein_set1) > 0:
                j = i + _j + 1
                adj[i][j] = 1
                adj[j][i] = 1
    D = nx.DiGraph(adj)
    edges = np.array([(u, v) for (u, v) in D.edges()])

    if s_path is not None and not os.path.exists(s_path):

        def adj_normalize(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -0.5).flatten()
            r_inv[np.isinf(r_inv)] = 0.0
            r_mat_inv = sps.diags(r_inv)
            mx = r_mat_inv.dot(mx).dot(r_mat_inv)
            return mx

        assert alpha is not None
        # Calculate S
        adj = sps.coo_matrix(adj)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        eigen_adj = alpha * np.linalg.inv((sps.eye(adj.shape[0]) - (1 - alpha) * adj_normalize(adj)).toarray())
        with open(s_path, "wb") as f:
            pickle.dump(eigen_adj, f)

    return edges
