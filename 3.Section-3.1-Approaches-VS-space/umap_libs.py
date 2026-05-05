#!/usr/bin/env python3
import argparse
import os
import pickle
import random
import time
import warnings
import numpy as np

try:
    import umap as umap_learn
    UMAP_LEARN_OK = True
except Exception:
    UMAP_LEARN_OK = False

try:
    from cuml.manifold import UMAP as cuUMAP
    CUML_OK = True
except Exception:
    CUML_OK = False

BASE_DIR = "../../../../Libraries"
PKL_NAME = "npy_medoids.pkl"
GLOBAL_CLUSTER_PKL = "global_medoids_bitbirch_count.pkl"

N_BITS = 2048
ACTIVE_LIB = "UF-Scripps-Actives"
DECOY_LIB = "UF-Scripps-Decoys"
AUTO_FORCE_CUML_N = 1000000


def set_global_seed(seed):
    """
    Set deterministic RNG state used by numpy/python consumers in this script.
    """
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_all_medoids(base_dir, pkl_name, exclude_libs=None):
    all_fps, all_sizes, all_labels, all_titles, all_smiles = [], [], [], [], []
    exclude_libs = set(exclude_libs or set())

    libraries = [
        d for d in sorted(os.listdir(base_dir))
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not libraries:
        raise RuntimeError(f"No subdirectories found under {base_dir}")

    for lib in libraries:
        if lib in exclude_libs:
            print(f"[SKIP] {lib} (excluded)")
            continue

        pkl_path = os.path.join(base_dir, lib, pkl_name)
        if not os.path.isfile(pkl_path):
            print(f"[WARN] {pkl_path} not found, skipping")
            continue

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        fps = np.asarray(data["fingerprints"], dtype=np.uint8)
        if fps.ndim != 2 or fps.shape[1] != N_BITS:
            raise ValueError(f"Unexpected fingerprint shape for {lib}: {fps.shape}")

        if "cluster_size" in data:
            sizes = np.asarray(data["cluster_size"], dtype=np.int64)
        elif "cluster_sizes" in data:
            sizes = np.asarray(data["cluster_sizes"], dtype=np.int64)
        else:
            raise KeyError(f"{pkl_path} missing cluster_size(s)")

        if fps.shape[0] != sizes.shape[0]:
            raise ValueError(f"Length mismatch in {lib}: {fps.shape[0]} vs {sizes.shape[0]}")

        labels = np.array([lib] * fps.shape[0], dtype=object)
        titles = np.asarray(data.get("titles_medoids", [None] * fps.shape[0]), dtype=object)
        smiles = np.asarray(data.get("smiles_medoids", [None] * fps.shape[0]), dtype=object)

        all_fps.append(fps)
        all_sizes.append(sizes)
        all_labels.append(labels)
        all_titles.append(titles)
        all_smiles.append(smiles)

    if not all_fps:
        raise RuntimeError("No medoids loaded")

    X = np.concatenate(all_fps, axis=0)
    sizes = np.concatenate(all_sizes, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    titles = np.concatenate(all_titles, axis=0)
    smiles = np.concatenate(all_smiles, axis=0)

    print(f"Loaded medoids: {X.shape[0]:,}")
    return X, sizes, labels, titles, smiles


def run_umap(
    X,
    n_neighbors=50,
    backend="auto",
    auto_force_cuml_n=AUTO_FORCE_CUML_N,
    rng_seed=52,
):
    backend = str(backend).lower()
    rng_seed = int(rng_seed)

    if backend == "auto" and X.shape[0] >= int(auto_force_cuml_n) and CUML_OK:
        print(
            f"[INFO] Auto backend: dataset size {X.shape[0]:,} >= {int(auto_force_cuml_n):,}; "
            "using cuML cosine for scalability."
        )
        backend = "cuml-cosine"

    if backend in {"auto", "umap-jaccard"}:
        if UMAP_LEARN_OK:
            X_bool = X.astype(bool, copy=False)
            retry_seeds = [rng_seed, rng_seed + 15, rng_seed + 27]
            last = None

            for seed in retry_seeds:
                reducer = umap_learn.UMAP(
                    n_neighbors=n_neighbors,
                    n_components=2,
                    metric="jaccard",
                    min_dist=0.05,
                    spread=1.0,
                    repulsion_strength=1.0,
                    negative_sample_rate=5,
                    set_op_mix_ratio=0.9,
                    local_connectivity=1.0,
                    init="spectral",
                    random_state=seed,
                    n_jobs=1,
                    verbose=True,
                )

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    emb = reducer.fit_transform(X_bool)

                nn_warn = any("Failed to correctly find n_neighbors" in str(x.message) for x in w)
                run_backend = f"umap-learn:jaccard:seed={seed}"
                last = (emb, run_backend, reducer)

                if not nn_warn:
                    print(f"UMAP Jaccard completed without NN warning (seed={seed})")
                    return emb, run_backend, reducer

                print(f"[WARN] NN warning with seed={seed}; retrying")

            print("[WARN] NN warning persisted for all retries; using last run")
            return last[0], last[1], last[2]

        if backend == "umap-jaccard":
            raise RuntimeError("Requested --backend umap-jaccard, but umap-learn is unavailable")

    # explicit or fallback cuML cosine
    if backend in {"auto", "cuml-cosine"}:
        if CUML_OK:
            print("Using cuML UMAP with cosine metric")
            reducer = cuUMAP(
                n_neighbors=n_neighbors,
                n_components=2,
                metric="cosine",
                min_dist=0.05,
                spread=1.0,
                repulsion_strength=1.0,
                negative_sample_rate=5,
                set_op_mix_ratio=0.9,
                local_connectivity=1.0,
                init="spectral",
                random_state=rng_seed,
                build_algo="nn_descent",
                build_kwds={"nnd_n_clusters": 8, "nnd_overlap_factor": 2},
                verbose=True,
                output_type="numpy",
            )
            emb = reducer.fit_transform(X.astype(np.float32, copy=False))
            return emb, f"cuml:cosine:seed={rng_seed}", reducer

        raise RuntimeError("Requested cuML cosine backend, but cuML is unavailable")

    raise ValueError(f"Unknown backend: {backend}")


def build_dataset_for_approach(args):
    X_all, sizes_all, labels_all, titles_all, smiles_all = load_all_medoids(
        args.base_dir,
        args.pkl_name,
        exclude_libs=set(),
    )

    idx = np.arange(X_all.shape[0], dtype=np.int64)
    mass = sizes_all.copy()
    if args.approach == 2:
        mass[:] = 1
        mass_mode = "ones"
        density_mode = "count"
        description = "Direct UMAP on library medoids with unit medoid mass"
    elif args.approach == 3:
        mass_mode = "original"
        density_mode = "weighted_vs_mass"
        description = "Direct UMAP on library medoids with original represented mass"
    else:
        raise ValueError(f"Unsupported approach for this script: {args.approach}")

    meta = {
        "approach": int(args.approach),
        "mass_mode": mass_mode,
        "selection_density_mode": density_mode,
        "description": description,
    }
    return X_all[idx], mass[idx], labels_all[idx], titles_all[idx], smiles_all[idx], idx, meta


def main():
    ap = argparse.ArgumentParser(description="Generate direct-medoid UMAP embeddings for approach 2 or 3")
    ap.add_argument("--approach", type=int, choices=[2, 3], required=True)
    ap.add_argument("--base-dir", default=BASE_DIR)
    ap.add_argument("--pkl-name", default=PKL_NAME)
    ap.add_argument("--n-neighbors", type=int, default=50)
    ap.add_argument("--output-prefix", default=None)
    ap.add_argument("--backend", choices=["auto", "umap-jaccard", "cuml-cosine"], default="auto")
    ap.add_argument("--rng-seed", type=int, default=52, help="Base random seed for UMAP and retries")
    ap.add_argument("--auto-force-cuml-n", type=int, default=AUTO_FORCE_CUML_N,
                    help="In backend=auto, switch to cuML cosine when number of points >= this threshold")
    args = ap.parse_args()
    set_global_seed(args.rng_seed)

    X, mass, labels, titles, smiles, idx, meta = build_dataset_for_approach(args)
    emb, backend, _ = run_umap(
        X,
        n_neighbors=args.n_neighbors,
        backend=args.backend,
        auto_force_cuml_n=args.auto_force_cuml_n,
        rng_seed=args.rng_seed,
    )

    out_prefix = args.output_prefix or f"umap_approach_{args.approach}"
    out_npz = out_prefix + "_embedding.npz"

    np.savez_compressed(
        out_npz,
        embedding=emb,
        cluster_sizes=mass,
        labels=labels,
        titles=titles,
        smiles=smiles,
        source_indices=idx,
        umap_backend=np.array([backend], dtype=object),
        approach=np.array([args.approach], dtype=int),
        approach_meta=np.array([meta], dtype=object),
    )
    print(f"Saved embedding: {out_npz}")


if __name__ == "__main__":
    main()
