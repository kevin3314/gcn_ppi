import argparse

import pandas as pd


def main(args):
    """
    Suppose the tsv file has the following format:
    ENSEMBL:ENSG00000242268 0.0     0.0     0.0     0.0     0.0     0
    ENSEMBL:ENSG00000270112 0.0     124.973306483   0.0     0.0     0.0
    """
    df = pd.read_csv(args.src_path, delimiter="\t", header=None)
    normalized_df = (df - df.mean()) / df.std()
    normalized_df[0] = df[0]
    normalized_df.to_csv(args.dst_path, sep="\t", index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize numerical feature")
    parser.add_argument("src_path")
    parser.add_argument("dst_path")
    args = parser.parse_args()
    main(args)
