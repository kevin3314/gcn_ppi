import argparse
import json
from itertools import chain
from pathlib import Path

from Bio.PDB import PDBList

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("genename2emsembl_pdb_path", type=Path, help="Path to genename2emsembl_pdb.json.")
    parser.add_argument("pdb_dir", type=Path, help="Path to directory in which pdb files are stored.")
    args = parser.parse_args()

    genename2emsebl_pdb_path: Path = args.genename2emsembl_pdb_path
    pdb_dir: Path = args.pdb_dir
    with open(genename2emsebl_pdb_path, "r") as f:
        genename2emsebl_pdb = json.load(f)
    pdb_ids_list = [value["pdb_id"] for value in genename2emsebl_pdb.values() if value["pdb_id"] is not None]
    # Remove duplicates
    pdb_ids = list(set(chain(*pdb_ids_list)))

    client = PDBList(pdb=pdb_dir)
    client.download_pdb_files(pdb_ids)
