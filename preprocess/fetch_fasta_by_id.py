import json
import time
from pathlib import Path

import requests

RES_DIR = Path("../data/fasta")
genename2emsebl_pdb_json_path = Path("/data1/NLP_PPI/ppi_data/mm_data/genename2emsebl_pdb.json")
with open(genename2emsebl_pdb_json_path, "r") as f:
    genename2emsebl_pdb = json.load(f)
server = "https://rest.ensembl.org"

for key, value in genename2emsebl_pdb.items():
    if value["emsebl_id"] is not None:
        emsebl_id = value["emsebl_id"]
        ext = "/sequence/id/" + emsebl_id + "?"
        res_path = RES_DIR / f"{emsebl_id}"
        time.sleep(1)
        r = requests.get(server + ext, headers={"Content-Type": "text/plain"})

        if r.ok:
            with open(res_path, "w") as f:
                f.write(r.text)
                print(f"Write result to {res_path}")
        if not r.ok:
            print("Fail on: ", emsebl_id)
