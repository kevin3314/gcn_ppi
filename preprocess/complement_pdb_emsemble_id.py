import argparse
import json

import pandas as pd


def main(args):
    df = pd.read_csv(args.src_csv)
    with open(args.mapping_json) as f:
        mapping = json.load(f)

    genenames0 = df["GENE_NAME0"]
    genenames1 = df["GENE_NAME1"]

    # Fetch pdb ids
    pdb_ids0 = []
    pdb_ids1 = []
    for genename0 in genenames0:
        try:
            pdb_ids = mapping[genename0]["pdb_id"]
            if isinstance(pdb_ids, list):
                pdb_ids0.append(pdb_ids[0].lower())
            elif isinstance(pdb_ids, str):
                pdb_ids0.append(pdb_ids.lower())
            else:
                assert pdb_ids is None
                pdb_ids0.append(None)
        except (KeyError, TypeError):
            pdb_ids0.append(None)

    for genename1 in genenames1:
        try:
            pdb_ids = mapping[genename1]["pdb_id"]
            if isinstance(pdb_ids, list):
                pdb_ids1.append(pdb_ids[0].lower())
            elif isinstance(pdb_ids, str):
                pdb_ids1.append(pdb_ids.lower())
            else:
                assert pdb_ids is None
                pdb_ids1.append(None)
        except (KeyError, TypeError):
            pdb_ids1.append(None)

    # Fetch ensemble ids
    ensemble_ids0 = []
    ensemble_ids1 = []
    for genename0 in genenames0:
        try:
            ensemble_id = mapping[genename0]["emsebl_id"]
            if ensemble_id is None:
                ensemble_ids0.append(None)
                continue
            assert type(ensemble_id) == str, f"{type(ensemble_id)} is not str"
            ensemble_ids0.append(ensemble_id)
        except (KeyError, TypeError):
            ensemble_ids0.append(None)

    for genename1 in genenames1:
        try:
            ensemble_id = mapping[genename1]["emsebl_id"]
            if ensemble_id is None:
                ensemble_ids1.append(None)
                continue
            assert type(ensemble_id) == str
            ensemble_ids1.append(ensemble_id)
        except (KeyError, TypeError):
            ensemble_ids1.append(None)

    df["PDB_ID0"] = pdb_ids0
    df["PDB_ID1"] = pdb_ids1
    df["ENSEMBLE_ID0"] = ensemble_ids0
    df["ENSEMBLE_ID1"] = ensemble_ids1
    df.to_csv(args.dst_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complement pdb ids.")
    parser.add_argument("src_csv", help="Path to source csv file.")
    parser.add_argument("dst_csv", help="Path to target csv file.")
    parser.add_argument("mapping_json", help="Mapping gene to pdb id")
    args = parser.parse_args()

    main(args)
