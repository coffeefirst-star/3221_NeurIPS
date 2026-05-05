import os
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_NPZ = os.path.join(BASE_DIR, "all_library_descriptor_cache.npz")
SUBSET_CSV = os.path.join(BASE_DIR, "selected_decoys.csv")
SUBSET_SMILES_COL = "SMILES"
PLOT_OUT = os.path.join(BASE_DIR, "descriptor_distributions_all_libraries_vs_subset_zoomed.png")
DEVIATION_OUT = os.path.join(BASE_DIR, "descriptor_distribution_deviation.csv")

DESCRIPTORS = {
    "MW": lambda mol: Descriptors.MolWt(mol),
    "LogP": lambda mol: Crippen.MolLogP(mol),
    "RotB": lambda mol: Lipinski.NumRotatableBonds(mol),
}

DESCRIPTOR_ORDER = [
    ("Molecular Weight", "MW"),
    ("LogP", "LogP"),
    ("Rotatable Bonds", "RotB"),
]

# Set to a tuple like (100, 700) to force a fixed x-range for a descriptor.
MANUAL_XLIMS = {
    "MW": None,
    "LogP": None,
    "RotB": None,
}

MANUAL_YLIMS = {
    "MW": None,
    "LogP": (0, 0.5),
    "RotB": None,
}

# Used when MANUAL_XLIMS[key] is None.
PERCENTILE_XLIMS = {
    "MW": (0.25, 99.5),
    "LogP": (0.5, 99.5),
    "RotB": (0.25, 99.0),
}

N_BINS = {
    "MW": 100,
    "LogP": 100,
    "RotB": 50,
}


def load_descriptor_cache(path):
    data = np.load(path)
    return {key: data[key] for key in DESCRIPTORS}


def compute_descriptors_from_smiles(smiles_list):
    values = {key: [] for key in DESCRIPTORS}
    invalid = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            invalid += 1
            continue
        for key, fn in DESCRIPTORS.items():
            values[key].append(fn(mol))
    return {key: np.asarray(vals, dtype=np.float64) for key, vals in values.items()}, invalid


def get_xlim(lib_values, subset_values, key):
    manual = MANUAL_XLIMS.get(key)
    if manual is not None:
        return manual

    combined = np.concatenate([lib_values, subset_values])
    lo_q, hi_q = PERCENTILE_XLIMS[key]
    lo = np.quantile(combined, lo_q / 100.0)
    hi = np.quantile(combined, hi_q / 100.0)
    if key == "RotB":
        lo = max(0, np.floor(lo))
        hi = np.ceil(hi)
    return float(lo), float(hi)


def l1_hist_distance(reference_values, subset_values, bins):
    reference_counts, _ = np.histogram(reference_values, bins=bins)
    subset_counts, _ = np.histogram(subset_values, bins=bins)
    reference_dist = reference_counts / max(1, reference_counts.sum())
    subset_dist = subset_counts / max(1, subset_counts.sum())
    return float(np.sum(np.abs(reference_dist - subset_dist)))


def main():
    print("CACHE_NPZ =", CACHE_NPZ)
    print("CACHE_NPZ exists?", os.path.exists(CACHE_NPZ))
    print("SUBSET_CSV =", SUBSET_CSV)
    print("SUBSET_CSV exists?", os.path.exists(SUBSET_CSV))

    if not os.path.isfile(CACHE_NPZ):
        raise RuntimeError(f"Descriptor cache not found: {CACHE_NPZ}")
    if not os.path.isfile(SUBSET_CSV):
        raise RuntimeError(f"Subset CSV not found: {SUBSET_CSV}")

    all_library_desc = load_descriptor_cache(CACHE_NPZ)

    subset_df = pd.read_csv(SUBSET_CSV)
    if SUBSET_SMILES_COL not in subset_df.columns:
        raise KeyError(f"Subset CSV must contain column '{SUBSET_SMILES_COL}'")
    subset_desc, subset_invalid = compute_descriptors_from_smiles(subset_df[SUBSET_SMILES_COL].tolist())

    print(f"Subset rows: {len(subset_df):,}")
    print(f"Subset invalid SMILES: {subset_invalid:,}")

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk")

    n_plots = len(DESCRIPTOR_ORDER)
    n_cols = 3
    n_rows = ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    deviation_rows = []

    for plot_idx, (ax, (title, key)) in enumerate(zip(axes, DESCRIPTOR_ORDER)):
        x_lib = all_library_desc[key]
        x_subset = subset_desc[key]
        if x_lib.size == 0 or x_subset.size == 0:
            ax.axis("off")
            continue

        xlim = get_xlim(x_lib, x_subset, key)
        if key == "RotB":
            bins = np.arange(int(xlim[0]) - 0.5, int(xlim[1]) + 1.5, 1)
        else:
            bins = N_BINS[key]
        ax.hist(
            x_lib,
            bins=bins,
            range=xlim,
            density=True,
            alpha=0.35,
            linewidth=1.0,
            label="Full VS libraries",
        )
        ax.hist(
            x_subset,
            bins=bins,
            range=xlim,
            density=True,
            alpha=0.35,
            linewidth=1.0,
            label="Approach 3: UMAP",
        )

        ax.set_xlim(*xlim)
        ylim = MANUAL_YLIMS.get(key)
        if ylim is not None:
            ax.set_ylim(*ylim)
            ax.set_yticks(np.linspace(ylim[0], ylim[1], 4))
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(title)
        if plot_idx == 0:
            ax.set_ylabel("Relative frequency")
        else:
            ax.set_ylabel("")
        if key == "RotB":
            ax.set_xticks(np.arange(int(xlim[0]), int(xlim[1]) + 1, 1))
        print(f"{key} xlim = {xlim}")
        if key == "LogP":
            ymax = ax.get_ylim()[1]
            ax.set_yticks(np.arange(0, ymax + 0.25, 0.25))
        if np.isscalar(bins):
            hist_bins = np.linspace(xlim[0], xlim[1], int(bins) + 1)
        else:
            hist_bins = np.asarray(bins, dtype=np.float64)
        deviation_rows.append(
            {
                "descriptor": key,
                "l1_hist_distance": l1_hist_distance(x_lib, x_subset, hist_bins),
            }
        )

    for ax in axes[n_plots:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(PLOT_OUT, dpi=800)
    plt.close(fig)
    pd.DataFrame(deviation_rows).to_csv(DEVIATION_OUT, index=False)

    print(f"Saved plot: {PLOT_OUT}")
    print(f"Saved deviation summary: {DEVIATION_OUT}")


if __name__ == "__main__":
    main()
