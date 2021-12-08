import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


def main(args: argparse.Namespace):
    np.random.seed(42)
    df: DataFrame = pd.read_csv(args.csv_path)
    train_ratio = args.ratio[0]
    val_ratio = args.ratio[0] + args.ratio[1]
    train, validate, test = np.split(
        df.sample(frac=1, random_state=42), [int(train_ratio * len(df)), int(val_ratio * len(df))]
    )
    res_paths: List[Path] = [args.res_dir / f"{split}.csv" for split in ("train", "valid", "test")]
    for _df, res_path in zip([train, validate, test], res_paths):
        print(f"Save to {res_path}")
        _df.to_csv(res_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split csv into trian,val,test")
    parser.add_argument("csv_path", type=Path, help="Path to csv to split.")
    parser.add_argument(
        "ratio",
        type=lambda x: list(map(float, x.split(","))),
        default="0.8,0.1,0.1",
        help="Split ratio separeted by , (e.g. 0.9,0.05,0.05)",
    )
    parser.add_argument("res_dir", type=Path, help="Path to directory in which save result")
    args = parser.parse_args()
    main(args)
