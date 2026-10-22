#!/usr/bin/env python3
"""Convert SPMe CSVs from the old LGM50 row timing to the current one.

Old timing (datasets generated before the 2026-07-04 fix in LGM50's
sim_profile_LGM50.m, e.g. the fresh-cell ``data/spme_*.csv`` files): the
voltage/state in row k was produced by the current stored in row k-1.

Current timing (e.g. ``data/*_soh90.csv``): row 1 is the initial state with
current 0, and row k holds the state produced by the current in row k.

The conversion only shifts ``current_A`` down by one row (first row = 0); all
other columns are unchanged. This was verified exactly: shifting the old
degraded dataset (git commit 27f7649) reproduces the regenerated one
(commit e03499d) to machine precision.

Usage:
    python tools/realign_current_column.py data/spme_train_FUDS.csv data/spme_train_FUDS_aligned.csv
"""
import argparse

import numpy as np
import pandas as pd


def realign(df):
    out = df.copy()
    current = df["current_A"].to_numpy()
    out["current_A"] = np.r_[0.0, current[:-1]]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)
    realign(df).to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv} ({len(df)} rows; current_A shifted by one row)")


if __name__ == "__main__":
    main()
