import json
from logging import INFO, getLogger
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import Bio.PDB
import dotenv
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy.sparse import coo_matrix, save_npz
from scipy.spatial import distance
from tqdm import tqdm

import src.utils

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="preprocess.yaml")
def main(config: DictConfig):
    """Load csv and preprocess about pdb.
    Then load the allocations of amino acid and calculate adjacency matrix
    based on the distance between them.
    Also build the dictionary of amino acid appearing in training split.
    """
    src.utils.print_config(config, resolve=True)
    logger = getLogger(__file__)
    logger.setLevel(INFO)

    train_path = Path(config.train_path)
    valid_path = Path(config.valid_path)
    test_path = Path(config.test_path)
    pdb_root = Path(config.pdb_root)
    res_root = Path(config.res_root)
    thr = config.threshold

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    PDB_COLS = ["PDB_ID0", "PDB_ID1"]
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
    res_root.mkdir(exist_ok=True, parents=True)
    with open(res_root / "vocab.json", "w") as f:
        logger.info(f"Save vocab to {res_root / 'vocab.json'}")
        json.dump(vocab, f, indent=4)
    for pdb_id, unit_ids, adjacency_matrix in zip(all_pdb_ids, units_ids, adjacency_matrixes):
        ids_name, adj_name = f"{pdb_id}_ids.npy", f"{pdb_id}_adj.npz"
        logger.info(f"Save {ids_name} and {adj_name} to {res_root}")
        np.save(res_root / ids_name, unit_ids)
        save_npz(res_root / adj_name, adjacency_matrix)
    with open(res_root / "config.yaml", "w") as f:
        OmegaConf.save(config=config, f=f.name)


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
    units_list, _ = get_names_coordinates_for_pdbs(pdbs, pdb_root)
    units_vocab = {
        units: i for i, units in enumerate(set(unit for units in units_list for unit in units if unit is not None))
    }
    return units_vocab


def get_graph_for_pdbs(
    pdbs: List[str],
    vocab: Dict[str, int],
    thr: Optional[float] = 8,
    pdb_root: Optional[Union[Path, str]] = "data/pdb",
):
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
    for pdb_id, pdb_path in tqdm(zip(pdbs, pdb_paths), desc="Loading pdb files...", total=len(list(pdbs))):
        structure = parser.get_structure(pdb_id, pdb_path)
        model = structure[0]
        chains: List[Bio.PDB.Chain.Chain] = model.get_chains()
        residues: List[Bio.PDB.Residue.Residue] = [
            residue for chain in chains for residue in chain.get_residues() if residue.full_id[-1][0] == " "
        ]
        residue_names.append([residue.resname for residue in residues])
        residue_coordinates.append(np.array([get_residual_coord(residue) for residue in residues]))
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


if __name__ == "__main__":
    main()
