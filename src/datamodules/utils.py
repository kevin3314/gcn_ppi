import hashlib
import os
import pickle
from logging import INFO, getLogger
from numbers import Number
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import Bio.PDB
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.spatial import distance
from tqdm import tqdm

logger = getLogger(__file__)
logger.setLevel(INFO)


def construct_graph(
    train_path: Union[str, Path],
    valid_path: Union[str, Path],
    test_path: Union[str, Path],
    pdb_root: Union[str, Path],
    threshold: Number,
) -> Tuple[Dict[str, np.ndarray], Dict[str, coo_matrix], Dict[str, int]]:
    """Load csv and build graph from pdb.
    Then load the allocations of amino acid and calculate adjacency matrix
    based on the distance between them.
    Also build the dictionary of amino acid appearing in training split.
    """
    logger = getLogger(__file__)
    logger.setLevel(INFO)

    train_path = Path(train_path)
    valid_path = Path(valid_path)
    test_path = Path(test_path)
    pdb_root = Path(pdb_root)
    thr = threshold

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)
    PDB_COLS = ["PDB_ID0", "PDB_ID1"]

    # Look up cache
    ids_for_lookup = ""
    for df in [train_df, valid_df, test_df]:
        ids_for_lookup += "_".join(sorted(map(lambda x: str(x), list(set(df["PDB_ID0"].values)))))
    hash_id = hashlib.md5(ids_for_lookup.encode("utf-8")).hexdigest()
    hashed_path = Path(f"{os.environ.get('CACHE_ROOT')}/{hash_id}.pkl")
    if hashed_path.exists() and not os.environ.get("OVERWRITE_CACHE", False):
        logger.info("Found cache of graph data: %s", hashed_path)
        with open(f"{os.environ.get('CACHE_ROOT')}/{hash_id}.pkl", "rb") as f:
            pdbid2nodes, pdbid2adjs, vocab = pickle.load(f)
            return pdbid2nodes, pdbid2adjs, vocab

    if os.environ.get("OVERWRITE_CACHE", False):
        logger.info("Overwrite cache of graph data: %s", hashed_path)
    else:
        logger.info("No cache of graph data found: %s", hashed_path)
    logger.info("Building graph data...")
    train_pdb_ids: Set = set()
    for col in PDB_COLS:
        train_pdb_ids |= set(train_df[col].values)
    train_pdb_ids = {x for x in train_pdb_ids if x is not np.nan}

    all_pdb_ids: Set = set()
    for df in [train_df, valid_df, test_df]:
        for col in PDB_COLS:
            _se = set(df[col].values)
            all_pdb_ids |= _se
    all_pdb_ids = {x for x in all_pdb_ids if x is not np.nan}
    train_pdb_ids = list(train_pdb_ids)
    all_pdb_ids = list(all_pdb_ids)

    train_pdb_ids = [pdb for pdb in train_pdb_ids if Path(get_pdb_path(pdb, pdb_root)).exists()]
    all_pdb_ids = [pdb for pdb in all_pdb_ids if Path(get_pdb_path(pdb, pdb_root)).exists()]

    # We cannot access to valid/test data during training (i.e. building vocabulary).
    vocab: Dict[str, int] = get_unit_vocab_for_pdbs(train_pdb_ids, pdb_root)
    units_ids, adjacency_matrixes = get_graph_for_pdbs(all_pdb_ids, vocab, thr, pdb_root)

    pdbid2nodes = {pdb_id: unit_ids for pdb_id, unit_ids in zip(all_pdb_ids, units_ids)}
    pdbid2adjs = {pdb_id: adj for pdb_id, adj in zip(all_pdb_ids, adjacency_matrixes)}

    hashed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hashed_path, "wb") as f:
        pickle.dump((pdbid2nodes, pdbid2adjs, vocab), f)
    return pdbid2nodes, pdbid2adjs, vocab


def get_unit_vocab_for_pdbs(
    pdbs: List[str],
    pdb_root: Optional[Union[Path, str]] = "data/pdb",
) -> Dict[str, int]:
    """Get vocabulary for units in pdbs.

    Args:
        pdbs (List[str]): Pdb ids.
        pdb_root (Optional[Union[Path, str]], optional): Directory which contains pdb ids.
                                                         Defaults to "/data1/NLP_PPI/ppi_data/pdb".
    """
    units_list, _ = get_names_coordinates_for_pdbs(pdbs, pdb_root, return_coordinates=False)
    units_vocab = {
        units: i for i, units in enumerate(set(unit for units in units_list for unit in units if unit is not None))
    }
    return units_vocab


def get_graph_for_pdbs(
    pdbs: List[str],
    vocab: Dict[str, int],
    thr: Optional[float] = 8,
    pdb_root: Optional[Union[Path, str]] = "data/pdb",
) -> Tuple[List[np.ndarray], List[coo_matrix]]:
    """Get graph for pdbs.

    Args:
        pdbs (List[str]): Pdb ids.

    Returns:
        np.ndarray: Graph.
    """
    units_list, coordinates_list = get_names_coordinates_for_pdbs(pdbs, pdb_root)
    UNK = len(vocab)
    units_ids = [np.array([vocab.get(unit, UNK) for unit in units]) for units in units_list]
    adjacency_matrixes: List[coo_matrix] = tqdm(
        [get_adjacency_matrix(coordinates, thr) for coordinates in coordinates_list],
        desc="Get adjacency matrix...",
    )
    return units_ids, adjacency_matrixes


def get_pdb_path(
    pdb_id: str,
    pdb_root: Optional[Union[Path, str]] = "data/pdb",
) -> str:
    """Get pdb path.

    Args:
        pdb_id (str): Pdb id.

    Returns:
        Str: Structure name.
        Str: Pdb path.
    """
    dir_name = pdb_id[1:3].lower()
    pdb_path = Path(pdb_root) / dir_name / f"{pdb_id}.cif"
    return pdb_path


def get_names_coordinates_for_pdbs(
    pdbs: List[str],
    pdb_root: Optional[Union[Path, str]] = "data/pdb",
    return_coordinates: bool = True,
) -> Tuple[List[List[str]], List[np.ndarray]]:
    """Get names and coordinates for pdbs.
    Basically following https://yoshidabenjiro.hatenablog.com/entry/2020/01/15/171333.
    In practice, some residue does not subsume C["CA"].
    On such case, instead use mean of all atoms.

    Args:
        pdbs (List[str]): Pdb ids.

    Returns:
        List[List[str]]: List of names for each sample.
        np.ndarray: List of coordinates for each sample.
    """
    parser = Bio.PDB.MMCIFParser()

    def get_residual_coord(r: Bio.PDB.Residue.Residue):
        try:
            return r["CA"].coord
        except KeyError:  # If CA does not exist, return mean of all atoms.
            return sum(atom.coord for atom in r.get_atoms()) / len(list(r.get_atoms()))

    pdb_paths: List[Path] = [get_pdb_path(pdb, pdb_root) for pdb in pdbs]
    parser = Bio.PDB.MMCIFParser(QUIET=True)
    residue_names: List[List[str]] = []
    residue_coordinates: List[np.ndarray] = []
    names_append = residue_names.append
    coord_append = residue_coordinates.append
    for pdb_id, pdb_path in tqdm(zip(pdbs, pdb_paths), desc="Loading pdb files...", total=len(list(pdbs))):
        structure = parser.get_structure(pdb_id, pdb_path)
        model = structure[0]
        chains: List[Bio.PDB.Chain.Chain] = model.get_chains()
        residues: List[Bio.PDB.Residue.Residue] = [
            residue for chain in chains for residue in chain.get_residues() if residue.full_id[-1][0] == " "
        ]
        names_append([residue.resname for residue in residues])
        if return_coordinates:
            coord_append(np.array([get_residual_coord(residue) for residue in residues]))
    return residue_names, residue_coordinates


def get_adjacency_matrix(coords: np.ndarray, thr: float = 8) -> coo_matrix:
    """Get adjacency matrix.

    Args:
        units (Str): Units
        coordinates (np.ndarray): Coordinates of each atoms.

    Returns:
        coo_matrix: Adjacency matrix.
    """
    # coords = coords[:, None, :] - coords
    distance_map = distance.cdist(coords, coords, "euclidean")
    contact_map = (distance_map <= thr).astype(int)
    return coo_matrix(contact_map)
