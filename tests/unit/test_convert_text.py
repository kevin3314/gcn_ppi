import numpy as np

from src.datamodules.datasets.convert_text import build_edges_by_proteins


def test_build_edges_by_proteins():
    # 0 -> A, B, C
    # 1 -> A, X, Y
    # 2 -> A, A, B
    ids = ["0", "0", "0", "1", "1", "1", "2", "2", "2"]
    protein0s = ["A", "B", "C", "A", "X", "Y", "A", "A", "B"]
    protein1s = ["B", "C", "A", "X", "Y", "A", "A", "B", "A"]
    ref0s = ["0", "1", "2", "0", "1", "2", "0", "1", "2"]
    ref1s = ["1", "2", "0", "1", "2", "0", "1", "2", "0"]

    # contain_protein_sets: [{'C'}, {'A'}, {'B'}, {'Y'}, {'A'}, {'X'}, {'B'}, {'A'}, {'A'}]

    edges = build_edges_by_proteins(ids, protein0s, protein1s, ref0s, ref1s)
    assert (
        edges
        == np.array(
            [
                [1, 4],
                [1, 7],
                [1, 8],
                [2, 6],
                [4, 1],
                [4, 7],
                [4, 8],
                [6, 2],
                [7, 1],
                [7, 4],
                [7, 8],
                [8, 1],
                [8, 4],
                [8, 7],
            ]
        )
    ).all()
