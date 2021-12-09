import numpy as np

from src.datamodules.datasets.convert_text import build_edges_by_proteins


def test_build_edges_by_proteins():
    # 0 -> A, B, C
    # 1 -> A, X, Y
    # 2 -> A, A, B
    ids = ["0", "0", "0", "1", "1", "1"]
    protein0s = ["A", "B", "C", "A", "X", "Y"]
    protein1s = ["B", "C", "A", "X", "Y", "A"]

    # contain_protein_sets: [{'C'}, {'A'}, {'B'}, {'Y'}, {'A'}, {'X'}, {'B'}, {'A'}, {'A'}]

    edges = build_edges_by_proteins(ids, protein0s, protein1s)
    assert (
        edges
        == np.array(
            [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [0, 5],
                [1, 0],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [2, 0],
                [2, 1],
                [2, 3],
                [2, 4],
                [2, 5],
                [3, 0],
                [3, 1],
                [3, 2],
                [3, 4],
                [3, 5],
                [4, 0],
                [4, 1],
                [4, 2],
                [4, 3],
                [4, 5],
                [5, 0],
                [5, 1],
                [5, 2],
                [5, 3],
                [5, 4],
            ]
        )
    ).all()
