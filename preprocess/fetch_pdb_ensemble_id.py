import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import mygene
from tqdm import tqdm


def extract_ids_from_dict_for_hprd(
    result: List[Dict], *args
) -> Tuple[Dict[str, Dict[Optional[str], Optional[str]]], int, int]:
    """Return dictionary mapping from gene name to pdb and ensembl id in case of HPRD.
    Return the number of missing items as well.

    Args:
        result (List[Dict]): Results of query of mygene.

    Returns:
        Tuple[Dict[str, Dict[Optional[str], Optional[str]]], int, int]:
    """
    emseble_id_none_count = 0
    pdb_id_none_count = 0
    result_dict = defaultdict(dict)
    for d in result:
        gene_name = d["query"]
        # List or dict
        emsebl_id = None
        emsebl_part = d.get("ensembl", None)
        # Emsebl is dict or list of dict or none
        if isinstance(emsebl_part, list):
            emsebl_part = emsebl_part[0]

        if isinstance(emsebl_part, dict):
            emsebl_id = emsebl_part.get("gene", None)
            emsebl_id = emsebl_id if emsebl_id is not None else None
        else:
            assert emsebl_part is None
        pdb_id = d.get("pdb", None)
        result_dict[gene_name]["emsebl_id"] = emsebl_id
        result_dict[gene_name]["pdb_id"] = pdb_id
        if emsebl_id is None:
            emseble_id_none_count += 1
        if pdb_id is None:
            pdb_id_none_count += 1
    return result_dict, emseble_id_none_count, pdb_id_none_count


def extract_ids_from_dict_for_bioinfer(
    result: List[Dict], protein_names: List[str]
) -> Tuple[Dict[str, Dict[Optional[str], Optional[str]]], int, int]:
    """Return dictionary mapping from gene name to pdb and ensembl id in case of BioInfer.
    Return the number of missing items as well.

    Args:
        result (List[Dict]): Results of query of mygene.

    Returns:
        Tuple[Dict[str, Dict[Optional[str], Optional[str]]], int, int]:
    """

    def get_checker(hits: List[Dict]) -> Callable[[Dict], bool]:
        """Get check function after sweeping all items.
        Check function is called to determine whether the item is the representative.
        1. One with ensembl id AND pdb id
        2. One with pdb id
        3. One with ensembl id

        Args:
            hits (List[Dict]): List of items.

        Returns:
            Callable[[Dict], bool]: Function for checking whether item meets criteria.
        """
        flags = [False, False, False]
        for item in hits:
            # If either of the two is None, then skip it.
            if "pdb" in item and "ensembl" in item:
                flags[0] = True
            elif "pdb" in item:
                flags[1] = True
            elif "ensembl" in item:
                flags[2] = True
        if flags[0]:
            return lambda x: "pdb" in x and "ensembl" in x
        elif flags[1]:
            return lambda x: "pdb" in x
        elif flags[2]:
            return lambda x: "ensembl" in x
        else:
            return lambda x: False

    emseble_id_none_count = 0
    pdb_id_none_count = 0
    result_dict = defaultdict(dict)
    for protein_name, d in tqdm(zip(protein_names, result)):
        ensembl_id = None
        pdb_id = None

        # Items in hits are ordered by score.
        # The representative is chosen based on the following criteria:
        # If no item meets the above criteria, none is set.
        if d is not None:
            hits = d["hits"]
            checker_function = get_checker(hits)
            for item in hits:
                if not checker_function(item):
                    continue
                pdb_part = item.get("pdb", None)
                ensembl_part = item.get("ensembl", None)

                if isinstance(pdb_part, list):
                    pdb_id = pdb_part[0]
                elif isinstance(pdb_part, str):
                    pdb_id = pdb_part
                else:
                    assert pdb_part is None

                if ensembl_part is not None:
                    if isinstance(ensembl_part, list):
                        ensembl_part = ensembl_part[0]
                    ensembl_id = ensembl_part["gene"]
                else:
                    assert ensembl_part is None

                break

        result_dict[protein_name]["emsebl_id"] = ensembl_id
        result_dict[protein_name]["pdb_id"] = pdb_id
        if ensembl_id is None:
            emseble_id_none_count += 1
        if pdb_id is None:
            pdb_id_none_count += 1
    return result_dict, emseble_id_none_count, pdb_id_none_count


DATASET2EXTRACT_FUNC = {
    "hprd": extract_ids_from_dict_for_hprd,
    "bioinfer": extract_ids_from_dict_for_bioinfer,
}


def fetch_result_for_hprd(gene_name_list: List[str]) -> List[Dict]:
    mg = mygene.MyGeneInfo()
    out = mg.querymany(gene_name_list, scopes="symbol", fields="ensembl, pdb", species="human")
    return out


def fetch_result_for_bioinfer(protein_name_list: List[str]) -> List[Dict]:
    mg = mygene.MyGeneInfo()
    protein_name = None
    out = []
    append = out.append
    for protein_name in tqdm(protein_name_list):
        try:
            append(mg.query([protein_name], scopes="symbol", fields="ensembl, pdb", species="human"))
        # Some protein name raise exception (e.g. one with slash, serine/threonine kinase)
        except Exception:
            print("Error occured:", protein_name)
            append(None)
    return out


DATASET2FETCH_FUNC = {
    "hprd": fetch_result_for_hprd,
    "bioinfer": fetch_result_for_bioinfer,
}

WARNING_MSG_ON_BIOINFER = """
WARNING: gene name is not provided in the bioinfer dataset.
         thus, fetch pdb and ensemble id by protein name instead.
"""


def main(args: argparse.Namespace):
    name_file_path: Path = args.text_path
    res_path: Path = args.res_path
    if args.dataset == "bioinfer":
        print(WARNING_MSG_ON_BIOINFER)
    fetch_func = DATASET2FETCH_FUNC[args.dataset]
    extract_func = DATASET2EXTRACT_FUNC[args.dataset]

    with open(name_file_path) as f:
        name_list = f.read().splitlines()

    out = fetch_func(name_list)

    result_dict, emseble_id_none_count, pdb_id_none_count = extract_func(out, name_list)
    print(f"Example output: {out[0]}")
    print("#None in emseble_id ->", emseble_id_none_count, "/", len(name_list))
    print("#None in pdb_id ->", pdb_id_none_count, "/", len(name_list))

    with open(res_path, "w") as f:
        json.dump(result_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ensembl ids and pdb ids from gene names.")
    parser.add_argument("text_path", type=Path, help="Path to text file containing gene names.")
    parser.add_argument("res_path", type=Path, help="Path to write result json file.")
    parser.add_argument("dataset", choices=["hprd", "bioinfer"], help="Type of dataset.")
    args = parser.parse_args()
    main(args)
