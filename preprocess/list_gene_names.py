from pathlib import Path

import pandas as pd

filepath: Path = Path("../data/mm_data/HPRD50_multimodal_dataset_reshaped.csv")
filedir: Path = filepath.parent
dstpath: Path = filedir / "gene_names.txt"

df = pd.read_csv(filepath)

gene_names = list(set(x.strip() for x in df["GENE_NAME0"]) | set(x.strip() for x in df["GENE_NAME1"]))
with open(dstpath, "w") as f:
    f.write("\n".join(gene_names))
