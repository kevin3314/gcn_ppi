from preprocess_pdb import get_adjacency_matrix


def test_get_adjacency_matrix():
    coords = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]

    contact_map = get_adjacency_matrix(coords, thr=1).toarray()
    assert contact_map[0, 1] == 1
    assert contact_map[0, 2] == 1
    assert contact_map[0, 3] == 1
    assert contact_map[0, 4] == 0

    contact_map = get_adjacency_matrix(coords, thr=1.5).toarray()
    assert contact_map[0, 1] == 1
    assert contact_map[0, 2] == 1
    assert contact_map[0, 3] == 1
    assert contact_map[0, 4] == 1

    contact_map = get_adjacency_matrix(coords, thr=0.8).toarray()
    assert contact_map[0, 1] == 0
    assert contact_map[0, 2] == 0
    assert contact_map[0, 3] == 0
    assert contact_map[0, 4] == 0
