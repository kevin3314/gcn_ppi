from typing import Dict, List, Tuple

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
