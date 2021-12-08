import argparse
import json
from collections import defaultdict
from pathlib import Path

import mygene


def main(args: argparse.Namespace):
    gene_name_file_path: Path = args.text_path
    res_path: Path = args.res_path

    with open(gene_name_file_path) as f:
        gene_name_list = f.read().splitlines()

    mg = mygene.MyGeneInfo()
    out = mg.querymany(gene_name_list, scopes="symbol", fields="ensembl, pdb", species="human")
    # out = mg.getgenes(gene_name_list,"ensembl, pdb")

    emseble_id_none_count = 0
    pdb_id_none_count = 0

    print(out[0])
    result_dict = defaultdict(dict)
    for d in out:
        gene_name = d["query"]
        # List or dict
        emsebl_id = None
        emsebl_part = d.get("ensembl", None)
        # Emsebl is dict or list of dict or none
        if isinstance(emsebl_part, list):
            emsebl_part = emsebl_part[0]

        if isinstance(emsebl_part, dict):
            emsebl_id = emsebl_part.get("protein", None)
            emsebl_id = emsebl_id[0] if emsebl_id is not None else None
        else:
            assert emsebl_part is None
        pdb_id = d.get("pdb", None)
        result_dict[gene_name]["emsebl_id"] = emsebl_id
        result_dict[gene_name]["pdb_id"] = pdb_id
        if emsebl_id is None:
            emseble_id_none_count += 1
        if pdb_id is None:
            pdb_id_none_count += 1

    print("#None in emseble_id ->", emseble_id_none_count, "/", len(gene_name_list))
    print("#None in pdb_id ->", pdb_id_none_count, "/", len(gene_name_list))

    with open(res_path, "w") as f:
        json.dump(result_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ensembl ids and pdb ids from gene names.")
    parser.add_argument("text_path", type=Path, help="Path to text file containing gene names.")
    parser.add_argument("res_path", type=Path, help="Path to write result json file.")
    args = parser.parse_args()
    main(args)
