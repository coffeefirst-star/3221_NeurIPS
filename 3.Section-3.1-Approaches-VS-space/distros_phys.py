import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../Libraries"))
SUBSET_CSV = os.path.join(BASE_DIR, "selected_decoys.csv")
SUBSET_SMILES_COL = "SMILES"
N_JOBS = max(1, min(12, (os.cpu_count() or 1)))
FORCE_REBUILD_CACHE = False
INTERNAL_PARALLEL_LIBRARIES = {"Mcule", "WuXi"}
INTERNAL_CHUNK_SIZE = 500000

PLOT_OUT = "descriptor_distributions_all_libraries_vs_subset.png"
SUMMARY_OUT = "descriptor_distribution_summary.csv"
LIBRARY_CACHE_OUT = "all_library_descriptor_cache.npz"
LIBRARY_SUMMARY_CACHE_OUT = "all_library_processing_summary.csv"

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


def load_smiles_for_library(lib_dir):
    smiles = []

    smi_files = sorted(glob.glob(os.path.join(lib_dir, "*.smi")))
    for path in smi_files:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                smiles.append(line.split()[0])

    sdf_files = sorted(glob.glob(os.path.join(lib_dir, "*.sdf")))
    for path in sdf_files:
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        for mol in suppl:
            if mol is None:
                continue
            smiles.append(Chem.MolToSmiles(mol, isomericSmiles=True))

    return smiles


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


def compute_descriptor_chunk(smiles_chunk):
    return compute_descriptors_from_smiles(smiles_chunk)


def chunked(seq, size):
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


def process_library(lib_dir):
    lib_name = os.path.basename(lib_dir)
    smiles = load_smiles_for_library(lib_dir)

    if lib_name in INTERNAL_PARALLEL_LIBRARIES and len(smiles) > INTERNAL_CHUNK_SIZE:
        chunks = list(chunked(smiles, INTERNAL_CHUNK_SIZE))
        max_workers = min(len(chunks), max(1, N_JOBS))
        print(
            f"[CHUNK] {lib_name} | molecules={len(smiles):,} | "
            f"chunks={len(chunks)} | chunk_size={INTERNAL_CHUNK_SIZE:,} | workers={max_workers}"
        )
        chunk_descs = []
        invalid = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(compute_descriptor_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                desc_chunk, invalid_chunk = future.result()
                chunk_descs.append(desc_chunk)
                invalid += invalid_chunk
        desc = concat_descriptor_dicts(chunk_descs)
    else:
        desc, invalid = compute_descriptors_from_smiles(smiles)

    summary = {
        "library": lib_name,
        "n_smiles_loaded": int(len(smiles)),
        "n_invalid_smiles": int(invalid),
        "n_valid_molecules": int(desc["MW"].shape[0]),
    }
    print(
        f"[DONE] {lib_name} | loaded={summary['n_smiles_loaded']:,} | "
        f"valid={summary['n_valid_molecules']:,} | invalid={summary['n_invalid_smiles']:,}"
    )
    return lib_name, desc, summary


def concat_descriptor_dicts(dicts):
    out = {key: [] for key in DESCRIPTORS}
    for d in dicts:
        for key in DESCRIPTORS:
            out[key].append(d[key])
    return {
        key: np.concatenate(vals, axis=0) if vals else np.array([], dtype=np.float64)
        for key, vals in out.items()
    }


def save_descriptor_cache(path, descriptor_dict):
    np.savez_compressed(path, **descriptor_dict)


def load_descriptor_cache(path):
    data = np.load(path)
    return {key: data[key] for key in DESCRIPTORS}


def summarize_descriptor_set(name, descriptor_dict):
    rows = []
    for key in DESCRIPTORS:
        arr = descriptor_dict[key]
        if arr.size == 0:
            rows.append(
                {
                    "dataset": name,
                    "descriptor": key,
                    "n": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "q05": np.nan,
                    "q95": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }
            )
            continue
        rows.append(
            {
                "dataset": name,
                "descriptor": key,
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "q05": float(np.quantile(arr, 0.05)),
                "q95": float(np.quantile(arr, 0.95)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        )
    return rows


def main():
    print("LIB_DIR =", LIB_DIR)
    print("LIB_DIR exists?", os.path.exists(LIB_DIR))
    print("SUBSET_CSV =", SUBSET_CSV)
    print("SUBSET_CSV exists?", os.path.exists(SUBSET_CSV))
    print("LIBRARY_CACHE_OUT =", LIBRARY_CACHE_OUT)
    print("FORCE_REBUILD_CACHE =", FORCE_REBUILD_CACHE)

    if not os.path.isdir(LIB_DIR):
        raise RuntimeError(f"Library directory not found: {LIB_DIR}")
    if not os.path.isfile(SUBSET_CSV):
        raise RuntimeError(f"Subset CSV not found: {SUBSET_CSV}")

    library_dirs = [
        os.path.join(LIB_DIR, d)
        for d in sorted(os.listdir(LIB_DIR))
        if os.path.isdir(os.path.join(LIB_DIR, d))
    ]
    print(f"Found {len(library_dirs)} library directories")
    print(f"Running with N_JOBS={N_JOBS}")

    if os.path.isfile(LIBRARY_CACHE_OUT) and not FORCE_REBUILD_CACHE:
        print(f"Loading cached all-library descriptors from: {LIBRARY_CACHE_OUT}")
        all_library_desc = load_descriptor_cache(LIBRARY_CACHE_OUT)
        if os.path.isfile(LIBRARY_SUMMARY_CACHE_OUT):
            library_summary_df = pd.read_csv(LIBRARY_SUMMARY_CACHE_OUT)
        else:
            library_summary_df = pd.DataFrame()
    else:
        library_descs = []
        library_summaries = []
        with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
            futures = {executor.submit(process_library, lib_dir): lib_dir for lib_dir in library_dirs}
            for future in as_completed(futures):
                lib_name, desc, summary = future.result()
                library_descs.append(desc)
                library_summaries.append(summary)

        all_library_desc = concat_descriptor_dicts(library_descs)
        save_descriptor_cache(LIBRARY_CACHE_OUT, all_library_desc)
        library_summary_df = pd.DataFrame(library_summaries).sort_values("n_valid_molecules", ascending=False)
        library_summary_df.to_csv(LIBRARY_SUMMARY_CACHE_OUT, index=False)
        print(f"Saved all-library cache: {LIBRARY_CACHE_OUT}")
        print(f"Saved library processing summary: {LIBRARY_SUMMARY_CACHE_OUT}")

    subset_df = pd.read_csv(SUBSET_CSV)
    if SUBSET_SMILES_COL not in subset_df.columns:
        raise KeyError(f"Subset CSV must contain column '{SUBSET_SMILES_COL}'")
    subset_desc, subset_invalid = compute_descriptors_from_smiles(subset_df[SUBSET_SMILES_COL].tolist())

    print(f"Subset molecules loaded: {len(subset_df):,}")
    print(f"Subset valid molecules: {subset_desc['MW'].shape[0]:,}")
    print(f"Subset invalid molecules: {subset_invalid:,}")

    summary_rows = []
    summary_rows.extend(summarize_descriptor_set("all_libraries", all_library_desc))
    summary_rows.extend(summarize_descriptor_set("subset", subset_desc))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk")

    n_plots = len(DESCRIPTOR_ORDER)
    n_cols = 2
    n_rows = ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (title, key) in zip(axes, DESCRIPTOR_ORDER):
        x_lib = all_library_desc[key]
        x_subset = subset_desc[key]

        if x_lib.size == 0 or x_subset.size == 0:
            ax.axis("off")
            continue

        sns.kdeplot(x=x_lib, ax=ax, fill=True, alpha=0.35, linewidth=1.2, label="All libraries")
        sns.kdeplot(x=x_subset, ax=ax, fill=True, alpha=0.35, linewidth=1.2, label="Created dataset")

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(title)
        ax.set_ylabel("Density")

    for ax in axes[n_plots:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(PLOT_OUT, dpi=600)
    plt.close(fig)

    if not library_summary_df.empty:
        print("\nTop libraries by valid molecule count:")
        print(library_summary_df.head(10).to_string(index=False))
    print(f"\nSaved descriptor summary: {SUMMARY_OUT}")
    print(f"Saved plot: {PLOT_OUT}")


if __name__ == "__main__":
    main()
