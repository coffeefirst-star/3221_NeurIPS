import os
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import SDMolSupplier, SanitizeMol, rdFingerprintGenerator


LIBRARIES_DIR = "../../../../Libraries"
PKL_NAME = "npy_medoids.pkl"

N_BITS = 2048
RADIUS = 2
CHUNK_SIZE = 10000
N_JOBS = max(1, min(32, (os.cpu_count() or 1)))

EXCLUDED_LIBS = {"UF-Scripps-Actives", "UF-Scripps-Decoys"}

SUBSET_CSV_PATH = "midcap_3_capmax_20_alpha_0.5_decoys.csv"
SUBSET_PKL_PATH = None
SUBSET_SDF_PATHS = None
SUBSET_SMILES_COL = "SMILES"
SUBSET_SANITIZE = "all"
SUBSET_SKIP_INVALID = True

HIST_OUT = "nn_similarity_histogram.png"
SUMMARY_OUT = "nn_similarity_summary.csv"
LIBRARY_SUMMARY_OUT = "nn_similarity_by_library.csv"
EXACT_MATCHES_OUT = "nn_tanimoto_1_pairs.csv"
EXACT_MATCH_LIBRARY = "Mcule"
SAVE_PER_MEDOID = False
PER_MEDOID_OUT = "nn_similarity_per_medoid.csv.gz"


def ensure_unpacked_fps(fps, n_bits=2048):
    fps = np.asarray(fps)
    if fps.ndim != 2:
        raise ValueError(f"Expected 2D fingerprint array, got shape {fps.shape}")

    packed_nbytes = n_bits // 8
    print(
        f"[FP PREP] raw fps shape={fps.shape}, dtype={fps.dtype}, "
        f"expected bits={n_bits}, expected packed bytes={packed_nbytes}"
    )

    if fps.shape[1] == n_bits:
        fps = fps.astype(np.uint8, copy=False)
        print(f"[FP PREP] fingerprints already unpacked -> shape={fps.shape}, dtype={fps.dtype}")
        return np.ascontiguousarray(fps)

    if fps.shape[1] == packed_nbytes and fps.dtype == np.uint8:
        unpacked = np.unpackbits(np.ascontiguousarray(fps), axis=1)[:, :n_bits].astype(np.uint8, copy=False)
        print(
            f"[FP PREP] packed fingerprints detected -> "
            f"input shape={fps.shape}, unpacked shape={unpacked.shape}, dtype={unpacked.dtype}"
        )
        return np.ascontiguousarray(unpacked)

    raise ValueError(
        f"Cannot interpret fingerprint shape {fps.shape} for n_bits={n_bits}. "
        f"Expected second dim = {n_bits} or {packed_nbytes}."
    )


def _get_fp_generator(radius=2, n_bits=2048):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def _get_sanitize_flags(sanitize="all"):
    if sanitize == "all":
        return Chem.SanitizeFlags.SANITIZE_ALL
    if sanitize == "none":
        return Chem.SanitizeFlags.SANITIZE_NONE
    raise ValueError(f"Unsupported sanitize option: {sanitize}")


def fps_from_mols(
    mols,
    radius=2,
    n_features=2048,
    dtype=np.uint8,
    sanitize="all",
    skip_invalid=False,
):
    if n_features < 1:
        raise ValueError("n_features must be greater than 0")

    print(
        f"[FP BUILD] molecules={len(mols)}, radius={radius}, n_features={n_features}, "
        f"dtype={np.dtype(dtype)}, sanitize={sanitize}, skip_invalid={skip_invalid}"
    )
    fpg = _get_fp_generator(radius=radius, n_bits=n_features)
    sanitize_flags = _get_sanitize_flags(sanitize)
    fps = np.empty((len(mols), n_features), dtype=dtype)
    invalid_idxs = []

    for i, mol in enumerate(mols):
        if mol is None:
            if skip_invalid:
                invalid_idxs.append(i)
                continue
            raise ValueError(f"Unable to parse molecule at idx {i} (None)")
        try:
            SanitizeMol(mol, sanitizeOps=sanitize_flags)
            arr = np.zeros((n_features,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fpg.GetFingerprint(mol), arr)
            fps[i, :] = arr
        except Exception:
            if skip_invalid:
                invalid_idxs.append(i)
                continue
            raise

    if invalid_idxs:
        fps = np.delete(fps, invalid_idxs, axis=0)
    fps = np.ascontiguousarray(fps, dtype=np.uint8)
    print(
        f"[FP BUILD] unpacked fingerprints shape={fps.shape}, dtype={fps.dtype}, "
        f"invalid_removed={len(invalid_idxs)}"
    )
    return fps, np.array(invalid_idxs, dtype=np.int64)


def smiles_to_unpacked_ecfp(smiles_list, radius=2, n_bits=2048, sanitize="all", skip_invalid=True):
    mols = []
    original_indices = []
    for i, smi in enumerate(smiles_list):
        if pd.isna(smi):
            if skip_invalid:
                continue
            raise ValueError(f"Missing SMILES at idx {i}")
        mols.append(Chem.MolFromSmiles(str(smi), sanitize=False))
        original_indices.append(i)

    if not mols:
        raise RuntimeError("No valid SMILES found in subset CSV.")

    fps, invalid_local_idxs = fps_from_mols(
        mols,
        radius=radius,
        n_features=n_bits,
        sanitize=sanitize,
        skip_invalid=skip_invalid,
    )
    invalid_local_idxs = set(invalid_local_idxs.tolist())
    valid_idx = np.asarray(
        [original_indices[i] for i in range(len(original_indices)) if i not in invalid_local_idxs],
        dtype=np.int64,
    )
    return fps, valid_idx


def fps_from_sdfs(sdf_paths, radius=2, n_features=2048, sanitize="all", skip_invalid=True):
    if isinstance(sdf_paths, (str, Path)):
        sdf_paths = [sdf_paths]
    mols = []
    for sdf_path in sdf_paths:
        suppl = SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
        mols.extend(list(suppl))
    return fps_from_mols(
        mols,
        radius=radius,
        n_features=n_features,
        sanitize=sanitize,
        skip_invalid=skip_invalid,
    )


def discover_library_pkls(libraries_dir, pkl_name, excluded_libs=None):
    if excluded_libs is None:
        excluded_libs = set()

    library_entries = []
    for lib in sorted(os.listdir(libraries_dir)):
        lib_dir = os.path.join(libraries_dir, lib)
        if not os.path.isdir(lib_dir):
            continue
        if lib in excluded_libs:
            print(f"[SKIP] {lib} (excluded from VS reference)")
            continue
        pkl_path = os.path.join(lib_dir, pkl_name)
        if os.path.isfile(pkl_path):
            library_entries.append((lib, pkl_path))

    if not library_entries:
        raise RuntimeError(f"No VS libraries loaded from {libraries_dir}")

    print("\nDiscovered VS libraries:", [lib for lib, _ in library_entries])
    return library_entries


def load_subset_from_pkl(pkl_path, n_bits=2048):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"[LOAD SUBSET PKL] keys: {list(data.keys())}")
    fps = ensure_unpacked_fps(data["fingerprints"], n_bits=n_bits)
    print(f"[LOAD SUBSET PKL] unpacked fps shape={fps.shape}, dtype={fps.dtype}")

    if "cluster_size" in data:
        weights = np.asarray(data["cluster_size"], dtype=np.float64)
    elif "cluster_sizes" in data:
        weights = np.asarray(data["cluster_sizes"], dtype=np.float64)
    else:
        weights = np.ones(fps.shape[0], dtype=np.float64)
    print(f"[LOAD SUBSET PKL] weights shape={weights.shape}, dtype={weights.dtype}")

    if fps.shape[0] != weights.shape[0]:
        raise ValueError(f"Subset PKL mismatch: fingerprints={fps.shape[0]}, weights={weights.shape[0]}")

    subset_df = pd.DataFrame({"source_index": np.arange(fps.shape[0], dtype=int)})
    return fps, weights, subset_df


def load_subset_from_source(csv_path=None, pkl_path=None, sdf_paths=None, n_bits=2048, radius=2):
    if pkl_path:
        return load_subset_from_pkl(pkl_path, n_bits=n_bits)

    if sdf_paths:
        fps, invalid_idxs = fps_from_sdfs(
            sdf_paths,
            radius=radius,
            n_features=n_bits,
            sanitize=SUBSET_SANITIZE,
            skip_invalid=SUBSET_SKIP_INVALID,
        )
        print(
            f"[LOAD SUBSET SDF] unpacked fps shape={fps.shape}, dtype={fps.dtype}, "
            f"invalid_removed={len(invalid_idxs)}"
        )
        subset_df = pd.DataFrame({"source_index": np.arange(fps.shape[0], dtype=int)})
        weights = np.ones(fps.shape[0], dtype=np.float64)
        return fps, weights, subset_df

    df = pd.read_csv(csv_path)
    if SUBSET_SMILES_COL not in df.columns:
        raise KeyError(f"Subset CSV must contain column '{SUBSET_SMILES_COL}'")

    fps, valid_idx = smiles_to_unpacked_ecfp(
        df[SUBSET_SMILES_COL].tolist(),
        radius=radius,
        n_bits=n_bits,
        sanitize=SUBSET_SANITIZE,
        skip_invalid=SUBSET_SKIP_INVALID,
    )
    print(
        f"[LOAD SUBSET CSV] unpacked fps shape={fps.shape}, dtype={fps.dtype}, "
        f"valid_rows={len(valid_idx)} / {len(df)}"
    )
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    weights = np.ones(fps.shape[0], dtype=np.float64)
    print(f"[LOAD SUBSET CSV] weights shape={weights.shape}, dtype={weights.dtype}")
    return fps, weights, df_valid


def nearest_neighbor_tanimoto(query_fps, candidate_fps, chunk_size=5000):
    if query_fps.ndim != 2 or candidate_fps.ndim != 2:
        raise ValueError("query_fps and candidate_fps must be 2D")
    if query_fps.shape[1] != candidate_fps.shape[1]:
        raise ValueError(
            f"Fingerprint dimension mismatch: {query_fps.shape[1]} vs {candidate_fps.shape[1]}"
        )

    query = np.ascontiguousarray(query_fps, dtype=np.uint8)
    cand = np.ascontiguousarray(candidate_fps, dtype=np.uint8)
    cand_t = cand.astype(np.uint16, copy=False).T
    cand_sum = cand.sum(axis=1, dtype=np.uint16)

    nn_scores = np.empty(query.shape[0], dtype=np.float32)
    nn_indices = np.empty(query.shape[0], dtype=np.int32)

    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        chunk = query[start:end].astype(np.uint16, copy=False)
        chunk_sum = chunk.sum(axis=1, dtype=np.uint16)
        inter = chunk @ cand_t
        union = chunk_sum[:, None] + cand_sum[None, :] - inter
        sims = np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0)
        best_idx = np.argmax(sims, axis=1)
        nn_indices[start:end] = best_idx.astype(np.int32, copy=False)
        nn_scores[start:end] = sims[np.arange(sims.shape[0]), best_idx].astype(np.float32, copy=False)
        print(
            f"[NN] processed {end:,} / {query.shape[0]:,} queries | "
            f"chunk mean nn tanimoto={nn_scores[start:end].mean():.4f}"
        )

    return nn_scores, nn_indices


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape[0] != weights.shape[0]:
        raise ValueError("values and weights must have same length")
    total_weight = weights.sum()
    if total_weight <= 0:
        return np.nan
    return float(np.dot(values, weights) / total_weight)


def summarize_nn_scores(nn_scores, weights=None):
    nn_scores = np.asarray(nn_scores, dtype=np.float64)
    weights = None if weights is None else np.asarray(weights, dtype=np.float64)
    return {
        "n_items": int(nn_scores.shape[0]),
        "mean_nn_tanimoto": float(nn_scores.mean()),
        "weighted_mean_nn_tanimoto": weighted_mean(nn_scores, weights) if weights is not None else np.nan,
        "median_nn_tanimoto": float(np.median(nn_scores)),
        "q05_nn_tanimoto": float(np.quantile(nn_scores, 0.05)),
        "q25_nn_tanimoto": float(np.quantile(nn_scores, 0.25)),
        "q75_nn_tanimoto": float(np.quantile(nn_scores, 0.75)),
        "q95_nn_tanimoto": float(np.quantile(nn_scores, 0.95)),
        "min_nn_tanimoto": float(nn_scores.min()),
        "max_nn_tanimoto": float(nn_scores.max()),
    }


def process_library_against_subset(library_name, pkl_path, subset_fps, n_bits=2048, chunk_size=5000):
    print(f"[LOAD VS] {library_name}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"[LOAD VS] {library_name} keys: {list(data.keys())}")
    fps = ensure_unpacked_fps(data["fingerprints"], n_bits=n_bits)
    print(f"[LOAD VS] {library_name} unpacked fps shape={fps.shape}, dtype={fps.dtype}")

    if "cluster_size" in data:
        cluster_size = np.asarray(data["cluster_size"], dtype=np.int64)
    elif "cluster_sizes" in data:
        cluster_size = np.asarray(data["cluster_sizes"], dtype=np.int64)
    else:
        raise KeyError(f"{pkl_path} missing cluster_size(s). Keys: {list(data.keys())}")
    print(f"[LOAD VS] {library_name} cluster_size shape={cluster_size.shape}, dtype={cluster_size.dtype}")

    medoid_indices = None
    medoid_smiles = None
    if "medoid_indices" in data:
        medoid_indices = np.asarray(data["medoid_indices"])
        print(f"[LOAD VS] {library_name} medoid_indices shape={medoid_indices.shape}, dtype={medoid_indices.dtype}")
        if medoid_indices.shape[0] != fps.shape[0]:
            raise ValueError(
                f"{library_name}: medoid_indices rows ({medoid_indices.shape[0]}) != fingerprints rows ({fps.shape[0]})"
            )
        if library_name == EXACT_MATCH_LIBRARY:
            if "medoid_smiles" in data:
                medoid_smiles = np.asarray(data["medoid_smiles"], dtype=object)
                print(f"[LOAD VS] {library_name} medoid_smiles shape={medoid_smiles.shape}, dtype={medoid_smiles.dtype}")
                if medoid_smiles.shape[0] != fps.shape[0]:
                    raise ValueError(
                        f"{library_name}: medoid_smiles rows ({medoid_smiles.shape[0]}) != fingerprints rows ({fps.shape[0]})"
                    )
            else:
                print(f"[LOAD VS] {library_name} has no medoid_smiles in PKL; exact-match SMILES will be unavailable")

    if fps.shape[0] != cluster_size.shape[0]:
        raise ValueError(
            f"{library_name}: fingerprints rows ({fps.shape[0]}) != cluster_size rows ({cluster_size.shape[0]})"
        )

    subset_scores, subset_best_idx = nearest_neighbor_tanimoto(
        query_fps=subset_fps,
        candidate_fps=fps,
        chunk_size=chunk_size,
    )

    summary = summarize_nn_scores(subset_scores)
    summary.update(
        {
            "library": library_name,
            "n_reference_medoids": int(fps.shape[0]),
            "represented_mass": float(cluster_size.sum()),
        }
    )
    return {
        "library": library_name,
        "subset_scores": subset_scores,
        "subset_best_idx": subset_best_idx,
        "medoid_indices": medoid_indices,
        "medoid_smiles": medoid_smiles,
        "library_summary": summary,
    }


def plot_nn_histogram(nn_scores, out_path):
    plt.figure(figsize=(8, 5))
    plt.hist(nn_scores, bins=80, color="#2c7fb8", alpha=0.9)
    plt.xlabel("Nearest-neighbor Tanimoto to reduced set")
    plt.ylabel("Reference medoid count")
    plt.title("Reference Medoids vs NN Tanimoto")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def build_final_best_by_subset(subset_scores_by_library, subset_weights=None):
    libraries = list(subset_scores_by_library.keys())
    score_matrix = np.column_stack([subset_scores_by_library[lib]["scores"] for lib in libraries])
    winning_col = np.argmax(score_matrix, axis=1)
    best_scores = score_matrix[np.arange(score_matrix.shape[0]), winning_col]
    best_libraries = np.asarray([libraries[idx] for idx in winning_col], dtype=object)
    best_ref_idx = np.asarray(
        [subset_scores_by_library[best_libraries[i]]["indices"][i] for i in range(len(best_libraries))],
        dtype=np.int64,
    )
    best_medoid_original_idx = []
    best_medoid_smiles = []
    for i, lib in enumerate(best_libraries):
        medoid_indices = subset_scores_by_library[lib]["medoid_indices"]
        medoid_smiles = subset_scores_by_library[lib]["medoid_smiles"]
        ref_idx = best_ref_idx[i]
        if medoid_indices is None:
            best_medoid_original_idx.append(np.nan)
        else:
            best_medoid_original_idx.append(medoid_indices[ref_idx])
        if medoid_smiles is None:
            best_medoid_smiles.append(np.nan)
        else:
            best_medoid_smiles.append(medoid_smiles[ref_idx])
    best_medoid_original_idx = np.asarray(best_medoid_original_idx, dtype=object)
    best_medoid_smiles = np.asarray(best_medoid_smiles, dtype=object)

    summary = summarize_nn_scores(best_scores, subset_weights)
    summary["n_subset_compounds"] = int(best_scores.shape[0])
    return best_scores, best_libraries, best_ref_idx, best_medoid_original_idx, best_medoid_smiles, summary


def summarize_winning_libraries(best_libraries, best_scores, subset_weights=None):
    best_libraries = np.asarray(best_libraries, dtype=object)
    best_scores = np.asarray(best_scores, dtype=np.float64)
    weights = np.ones(best_scores.shape[0], dtype=np.float64) if subset_weights is None else np.asarray(subset_weights, dtype=np.float64)

    rows = []
    for lib in np.unique(best_libraries):
        mask = best_libraries == lib
        lib_scores = best_scores[mask]
        lib_weights = weights[mask]
        rows.append(
            {
                "library": lib,
                "n_subset_wins": int(mask.sum()),
                "weighted_win_count": float(lib_weights.sum()),
                "mean_final_nn_tanimoto": float(lib_scores.mean()),
                "weighted_mean_final_nn_tanimoto": weighted_mean(lib_scores, lib_weights),
                "median_final_nn_tanimoto": float(np.median(lib_scores)),
                "q05_final_nn_tanimoto": float(np.quantile(lib_scores, 0.05)),
                "q95_final_nn_tanimoto": float(np.quantile(lib_scores, 0.95)),
            }
        )
    return pd.DataFrame(rows).sort_values("weighted_win_count", ascending=False).reset_index(drop=True)


def main():
    print("\n[STEP] Loading reduced subset...")
    subset_fps, subset_weights, subset_df = load_subset_from_source(
        csv_path=SUBSET_CSV_PATH,
        pkl_path=SUBSET_PKL_PATH,
        sdf_paths=SUBSET_SDF_PATHS,
        n_bits=N_BITS,
        radius=RADIUS,
    )

    print("\n[STEP] Discovering VS libraries...")
    library_entries = discover_library_pkls(
        libraries_dir=LIBRARIES_DIR,
        pkl_name=PKL_NAME,
        excluded_libs=EXCLUDED_LIBS,
    )

    print(f"\n[STEP] Computing per-library nearest-neighbor Tanimoto in parallel with N_JOBS={N_JOBS}...")
    library_results = {}
    library_summary_rows = []
    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        future_to_lib = {
            executor.submit(
                process_library_against_subset,
                lib,
                pkl_path,
                subset_fps,
                N_BITS,
                CHUNK_SIZE,
            ): lib
            for lib, pkl_path in library_entries
        }
        for future in as_completed(future_to_lib):
            lib = future_to_lib[future]
            result = future.result()
            library_results[lib] = {
                "scores": result["subset_scores"],
                "indices": result["subset_best_idx"],
                "medoid_indices": result["medoid_indices"],
                "medoid_smiles": result["medoid_smiles"],
            }
            library_summary_rows.append(result["library_summary"])
            print(
                f"[DONE] {lib} | mean nn={result['library_summary']['mean_nn_tanimoto']:.4f} | "
                f"subset size={subset_fps.shape[0]:,}"
            )

    best_scores, best_libraries, best_ref_idx, best_medoid_original_idx, best_medoid_smiles, summary = build_final_best_by_subset(
        library_results,
        subset_weights=subset_weights,
    )
    summary["loaded_libraries"] = ",".join(sorted(library_results.keys()))
    summary["subset_size"] = int(subset_fps.shape[0])
    summary["subset_weight_sum"] = float(np.sum(subset_weights))

    library_summary_df = pd.DataFrame(library_summary_rows).sort_values(
        "weighted_mean_nn_tanimoto", ascending=False
    ).reset_index(drop=True)
    winning_library_df = summarize_winning_libraries(
        best_libraries,
        best_scores,
        subset_weights=subset_weights,
    )
    winning_library_df.to_csv(LIBRARY_SUMMARY_OUT, index=False)
    pd.DataFrame([summary]).to_csv(SUMMARY_OUT, index=False)
    plot_nn_histogram(best_scores, HIST_OUT)

    exact_mask = np.isclose(best_scores, 1.0) & (best_libraries == EXACT_MATCH_LIBRARY)
    exact_df = pd.DataFrame(
        {
            "subset_idx": np.arange(best_scores.shape[0], dtype=np.int64),
            "best_library": best_libraries,
            "best_library_ref_idx": best_ref_idx,
            "best_library_medoid_index": best_medoid_original_idx,
            "best_library_medoid_smiles": best_medoid_smiles,
            "final_nn_tanimoto": best_scores,
            "subset_weight": subset_weights,
        }
    )
    if isinstance(subset_df, pd.DataFrame):
        subset_df_reset = subset_df.reset_index(drop=True).copy()
        exact_df = pd.concat([exact_df, subset_df_reset], axis=1)
    exact_df = exact_df.loc[exact_mask].reset_index(drop=True)
    exact_df.to_csv(EXACT_MATCHES_OUT, index=False)

    if SAVE_PER_MEDOID:
        out_df = pd.DataFrame(
            {
                "subset_idx": np.arange(best_scores.shape[0], dtype=np.int64),
                "best_library": best_libraries,
                "best_library_ref_idx": best_ref_idx,
                "final_nn_tanimoto": best_scores,
                "subset_weight": subset_weights,
            }
        )
        out_df.to_csv(PER_MEDOID_OUT, index=False)
        print(f"Saved per-medoid NN table: {PER_MEDOID_OUT}")

    print("\n================ SUMMARY ================")
    print(f"Reduced subset size:         {summary['subset_size']:,}")
    print(f"Mean final NN Tanimoto:      {summary['mean_nn_tanimoto']:.6f}")
    print(f"Weighted mean final NN:      {summary['weighted_mean_nn_tanimoto']:.6f}")
    print(f"Median final NN Tanimoto:    {summary['median_nn_tanimoto']:.6f}")
    print(f"5th / 95th percentile:       {summary['q05_nn_tanimoto']:.6f} / {summary['q95_nn_tanimoto']:.6f}")
    print(f"Min / Max final NN:          {summary['min_nn_tanimoto']:.6f} / {summary['max_nn_tanimoto']:.6f}")
    print(f"Saved summary CSV:           {SUMMARY_OUT}")
    print(f"Saved library summary CSV:   {LIBRARY_SUMMARY_OUT}")
    print(f"Saved exact-match CSV:       {EXACT_MATCHES_OUT}")
    print(f"Exact-match library filter:  {EXACT_MATCH_LIBRARY}")
    print(f"Saved histogram:             {HIST_OUT}")
    print("\nTop libraries by mean NN against subset:")
    print(library_summary_df.head(10).to_string(index=False))
    print("\nLibraries winning the final best-NN assignment:")
    print(winning_library_df.head(10).to_string(index=False))
    print("=========================================\n")


if __name__ == "__main__":
    main()
