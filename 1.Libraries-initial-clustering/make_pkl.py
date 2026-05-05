#!/usr/bin/env python3
import numpy as np
import pandas as pd
import bblean
import pickle
import os

# Input
CSV_PATH = "actives.csv"

# Output
OUTPUT_NAME = "npy_medoids.pkl"
N_FEATURES = 2048


def main():
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found")

    df = pd.read_csv(CSV_PATH)


    smiles = df["SMILES"].values.astype(str)
    titles = df["Title"].values.astype(str)

    print(f"Loaded {len(smiles)} actives.")

    # Unpacked fingerprints for UMAP (no clustering!)
    fps, invalid_idx = bblean.fps_from_smiles(
        smiles,
        pack=False,
        skip_invalid=True,
        n_features=N_FEATURES,
        kind="ecfp4",
    )

    # Remove invalid SMILES
    if len(invalid_idx) > 0:
        print(f"Skipping {len(invalid_idx)} invalid SMILES")
        mask = np.ones(len(smiles), dtype=bool)
        mask[invalid_idx] = False
        smiles = smiles[mask]
        titles = titles[mask]

    # Convert to uint8
    fps = fps.astype(np.uint8)

    # Each active is one cluster of size 1
    cluster_size = np.ones(len(fps), dtype=np.int32)
    medoid_indices = np.arange(len(fps), dtype=np.int64)

    # Save in BitBirch-compatible format
    with open(OUTPUT_NAME, "wb") as f:
        pickle.dump(
            {
                "fingerprints": fps,
                "cluster_size": cluster_size,
                "medoid_indices": medoid_indices,
                "titles_medoids": titles,
                "smiles_medoids": smiles,
            },
            f,
        )

    print(f"Saved {len(fps)} actives to {OUTPUT_NAME}")


if __name__ == "__main__":
    main()


