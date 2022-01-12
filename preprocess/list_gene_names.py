import argparse
from pathlib import Path

import pandas as pd


def main(args: argparse.Namespace):
    filepath: Path = args.csv_path
    dstpath: Path = args.dst_path

    df = pd.read_csv(filepath)

    gene_names = list(set(x.strip() for x in df["GENE_NAME0"]) | set(x.strip() for x in df["GENE_NAME1"]))
    with open(dstpath, "w") as f:
        f.write("\n".join(gene_names))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("csv_path", type=Path, help="Path to csv file to load.")
    parser.add_argument("dst_path", type=Path, help="Path to write result text file.")
    args = parser.parse_args()
    main(args)
