from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np


def convert_text_to_one_hot_vector(texts: List[str]) -> Tuple[Dict[str, int], List[np.ndarray]]:
    """Convert text to one hot vector with vocabulary.add()

    Args:
        text (List[str]): Text to process.

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

    return vocab, bag_of_words


def build_edges_by_proteins(ids: Sequence[str], protein0s: Sequence[int], pritein1s: Sequence[int]) -> np.ndarray:
    """Build edges based on whether instance shares same protein.

    MEMO:
        Convert protein name to index should boost performance.

    Args:
        ids (Sequence[int]): Ids of instances.
        protein0s (Sequence[int]): Protein ids of instance 0.
        pritein1s (Sequence[int]): Protein ids of instance 1.

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
    return edges
