#!/usr/bin/env python3
import argparse
import os
import pickle
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bblean.similarity as iSIM
from bblean.fingerprints import pack_fingerprints
from bblean.bitbirch import BitBirch

# ----------------------------
# Paths / IO
# ----------------------------
LIBRARIES_DIR = "../../../Libraries"
PKL_NAME = "npy_medoids.pkl"

OUT_PKL = "global_medoids_bitbirch_count.pkl"
OUT_NPY = "selected_medoids_idx.npy"
OUT_TXT = "selected_summary.txt"
OUT_SELECTED_CSV = "selected_decoys_benchmark.csv"
OUT_APPROACH3_ALLOC_CSV = "approach3_cluster_allocation.csv"
OUT_ACTIVE_CLUSTER_OVERLAP_CSV = "active_cluster_overlap_diagnostics.csv"

# ----------------------------
# Libraries
# ----------------------------
ACTIVE_LIB = "UF-Scripps-Actives"
DECOY_LIB  = "UF-Scripps-Decoys"

# Load all libraries with a PKL. Actives stay in the clustering pass so we can
# identify active-containing global clusters for the analog-decoy quota.
LOAD_ONLY_ACTIVES_AND_DECOYS = False
EXCLUDE_ACTIVES_FROM_CLUSTERING = False
EXCLUDE_DECOYS_FROM_CLUSTERING = True
# ----------------------------
# Fingerprint settings
# ----------------------------
N_BITS = 2048  # must match how your PKLs were generated
# ----------------------------
# Global clustering params
# ----------------------------
RNG_SEED = 42
BRANCHING_FACTOR = 1024
MERGE_CRITERION = "diameter"
RECLUSTER_ITERS = 8
THRESH_Z = 3.5          # threshold = mean + THRESH_Z * std
RECLUSTER_EXTRA = None  # if None -> use std from similarity sample


HARD_CAP_MIN_COUNT = 0   # set 1 if you want (almost) every nonempty cluster touched
HARD_CAP_MAX_COUNT = 1

# How to choose medoids within each global cluster to fill the count quota
# False -> greedy largest-mass-first (hexbin-like)
# True  -> weighted random ordering by mass (diversity-friendly)
WEIGHTED_ORDER_WITHIN_CLUSTER = False
SELECTION_APPROACH = 2  # 2=top-k one-per-cluster, 3=anchor+proportional
APPROACH3_ANCHOR_TOPK = 600
APPROACH3_POOL_TOPK = 3000
APPROACH3_WEIGHT_EXP = 1.0
ANALOG_FRACTION = 0.25
# ----------------------------
# Helpers
# ----------------------------
def set_global_seed(seed):
    """
    Set deterministic RNG state used by numpy/python consumers in this script.
    """
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)


def ensure_packed_fps(fps_u8, n_bits=N_BITS):
    """Accept unpacked (n, n_bits) or packed (n, n_bits//8) and return packed (uint8)."""
    fps_u8 = np.asarray(fps_u8, dtype=np.uint8)
    if fps_u8.ndim != 2:
        raise ValueError(f"fingerprints must be 2D, got shape {fps_u8.shape}")

    if fps_u8.shape[1] == n_bits:
        return pack_fingerprints(fps_u8)              # unpacked -> packed
    if fps_u8.shape[1] == (n_bits // 8):
        return fps_u8                                 # already packed

    raise ValueError(
        f"Unexpected fingerprint width {fps_u8.shape[1]}. "
        f"Expected {n_bits} (unpacked) or {n_bits//8} (packed)."
    )


def spearman_rho(x, y):
    """Spearman rho without SciPy (average ranks for ties)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def rankdata(a):
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)

        sorted_a = a[order]
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
                j += 1
            if j > i:
                avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
                for k in range(i, j + 1):
                    ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = rankdata(x)
    ry = rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    if denom == 0:
        return np.nan
    return float((rx * ry).sum() / denom)


def mass_coverage_topk_global(selected_idx, cluster_size, eligible_mask, K):
    """
    numerator: sum(mass[selected])
    denom: sum of top-K masses among eligible
    """
    cs = np.asarray(cluster_size, dtype=np.float64)
    selected_idx = np.asarray(selected_idx, dtype=int)
    eligible = np.where(np.asarray(eligible_mask, dtype=bool))[0]
    if eligible.size == 0 or K <= 0:
        return np.nan
    K_eff = min(int(K), int(eligible.size))

    eligible_mass = cs[eligible]
    if K_eff == eligible_mass.size:
        denom = float(eligible_mass.sum())
    else:
        denom = float(np.partition(eligible_mass, -K_eff)[-K_eff:].sum())

    num = float(cs[selected_idx].sum()) if selected_idx.size else 0.0
    print("denominator in mass (top-K eligible):", denom)
    print("numerator in mass (selected):", num)
    return num / denom if denom > 0 else np.nan


def read_active_K():
    """Get K = #active medoids from the actives PKL, without including actives in clustering."""
    pkl_path = os.path.join(LIBRARIES_DIR, ACTIVE_LIB, PKL_NAME)
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Missing actives PKL for K: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    fps = np.asarray(data["fingerprints"], dtype=np.uint8)
    return int(fps.shape[0])


def allocate_counts_with_capacity(theoretical, capacity, total_k):
    """
    Deterministic integer allocation with hard capacity and near-theoretical adherence.

    Steps:
    1) floor allocation
    2) distribute remaining units by largest fractional remainders
    3) never exceed ceil(theoretical) or capacity
    """
    theoretical = np.asarray(theoretical, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.int64)

    if theoretical.shape != capacity.shape:
        raise ValueError("theoretical and capacity must have same shape")
    if np.any(capacity < 0):
        raise ValueError("capacity must be non-negative")
    if total_k < 0:
        raise ValueError("total_k must be >= 0")

    # hard max per cluster from theory and physical availability
    max_allowed = np.minimum(np.ceil(np.clip(theoretical, 0.0, None)).astype(np.int64), capacity)
    if int(max_allowed.sum()) < int(total_k):
        raise RuntimeError(
            f"Cannot allocate {total_k} picks under theoretical/capacity limits "
            f"(max feasible={int(max_allowed.sum())})"
        )

    q = np.minimum(np.floor(np.clip(theoretical, 0.0, None)).astype(np.int64), max_allowed)
    need = int(total_k) - int(q.sum())
    if need <= 0:
        return q

    frac = np.clip(theoretical, 0.0, None) - np.floor(np.clip(theoretical, 0.0, None))
    eligible = np.where(q < max_allowed)[0]
    if eligible.size == 0:
        return q

    # Largest-remainder method with deterministic tie-break on index.
    # lexsort uses last key as primary, so (-frac) is primary and idx secondary.
    order = np.lexsort((eligible, -frac[eligible]))
    ranked = eligible[order]

    for idx in ranked:
        if need <= 0:
            break
        q[idx] += 1
        need -= 1

    if need > 0:
        raise RuntimeError(f"Allocation shortfall after remainder pass: remaining={need}")

    return q


def select_active_cluster_analogs(global_clusters, labels_all, cluster_sizes_all, active_lib, decoy_lib, analog_k):
    if analog_k <= 0:
        return np.array([], dtype=np.int64), np.zeros(len(global_clusters), dtype=np.int64)

    eligible = []
    for gid, gc in enumerate(global_clusters):
        members = gc["medoid_indices"]
        has_active = np.any(labels_all[members] == active_lib)
        has_decoy = np.any(labels_all[members] == decoy_lib)
        if has_active and has_decoy:
            eligible.append(gid)

    if not eligible:
        return np.array([], dtype=np.int64), np.zeros(len(global_clusters), dtype=np.int64)

    ranked = sorted(
        eligible,
        key=lambda gid: (
            int(global_clusters[gid]["represented_mass_all"]),
            int(global_clusters[gid].get("active_mass", 0)),
            int(global_clusters[gid].get("decoy_mass", 0)),
        ),
        reverse=True,
    )

    selected = []
    counts = np.zeros(len(global_clusters), dtype=np.int64)
    remaining = int(analog_k)

    for gid in ranked:
        if remaining <= 0:
            break
        members = global_clusters[gid]["medoid_indices"]
        dec_members = members[labels_all[members] == decoy_lib]
        if dec_members.size == 0:
            continue
        masses = cluster_sizes_all[dec_members].astype(np.int64)
        order = np.argsort(-masses, kind="mergesort")
        take = dec_members[order[: min(remaining, dec_members.size)]]
        if take.size:
            selected.append(take.astype(np.int64))
            counts[gid] += int(take.size)
            remaining -= int(take.size)

    if not selected:
        return np.array([], dtype=np.int64), counts

    return np.concatenate(selected).astype(np.int64), counts


def main():
    parser = argparse.ArgumentParser(description="Global inter-library reclustering + decoy selection")
    parser.add_argument("--approach", type=int, choices=[0, 2, 3], default=SELECTION_APPROACH,
                        help="0=cluster export only (no decoy clustering/selection), 2=top-k one-per-cluster, 3=mixed top-k")
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED,
                        help="Global random seed used for sampling/shuffling paths")
    parser.add_argument("--recluster-shuffle", type=int, choices=[0, 1], default=1,
                        help="BitBirch recluster_inplace shuffle flag (1=on, 0=off for max reproducibility)")
    parser.add_argument("--mix-topk-fraction", type=float, default=0.80,
                        help="Deprecated for approach 3 (kept for backward compatibility)")
    parser.add_argument("--approach3-anchor-topk", type=int, default=APPROACH3_ANCHOR_TOPK,
                        help="Approach 3: take 1 decoy from each of top-N clusters by VS mass")
    parser.add_argument("--approach3-pool-topk", type=int, default=APPROACH3_POOL_TOPK,
                        help="Approach 3: allocate remaining picks within top-M clusters by weighted repeated sampling")
    parser.add_argument("--approach3-weight-exp", type=float, default=APPROACH3_WEIGHT_EXP,
                        help="Approach 3: exponent on VS-mass weights during repeated sampling (1.0=linear)")
    parser.add_argument("--analog-fraction", type=float, default=ANALOG_FRACTION,
                        help="Fraction of K reserved for decoys drawn from the largest active-containing clusters")
    args = parser.parse_args()

    approach = int(args.approach)
    rng_seed = int(args.rng_seed)
    recluster_shuffle = bool(int(args.recluster_shuffle))
    set_global_seed(rng_seed)
    mix_frac = float(np.clip(args.mix_topk_fraction, 0.0, 1.0))
    approach3_anchor_topk = int(max(0, args.approach3_anchor_topk))
    approach3_pool_topk = int(max(1, args.approach3_pool_topk))
    approach3_weight_exp = float(max(0.0, args.approach3_weight_exp))
    analog_fraction = float(np.clip(args.analog_fraction, 0.0, 1.0))

    # Approach 0 is used as preprocessing for UMAP protocols 1/4
    exclude_decoys_from_clustering = (approach == 0)

    # ----------------------------
    # Load & concatenate medoids (decoys + vendor libs; optionally exclude actives)
    # ----------------------------
    all_fps_packed = []
    all_cluster_sizes = []
    all_labels = []
    loaded_libs = []
    lib_offsets = {}
    cursor = 0

    # For exporting selected decoys (we use titles/smiles from the DECOY_LIB PKL if present)
    all_titles = []
    all_smiles = []

    for lib in sorted(os.listdir(LIBRARIES_DIR)):
        lib_dir = os.path.join(LIBRARIES_DIR, lib)
        if not os.path.isdir(lib_dir):
            continue

        if LOAD_ONLY_ACTIVES_AND_DECOYS and lib not in {ACTIVE_LIB, DECOY_LIB}:
            continue

        # Skip actives/decoys from global inter-library clustering if requested
        if EXCLUDE_ACTIVES_FROM_CLUSTERING and lib == ACTIVE_LIB:
            print(f"[SKIP] {lib} (excluded from clustering)")
            continue
        if exclude_decoys_from_clustering and lib == DECOY_LIB:
            print(f"[SKIP] {lib} (excluded from clustering)")
            continue

        pkl_path = os.path.join(lib_dir, PKL_NAME)
        if not os.path.isfile(pkl_path):
            continue

        print(f"[LOAD] {lib}")
        loaded_libs.append(lib)

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # fingerprints can be packed or unpacked depending on how you created PKLs
        fps = ensure_packed_fps(data["fingerprints"], n_bits=N_BITS)

        if "cluster_size" in data:
            cs = np.asarray(data["cluster_size"], dtype=np.int64)
        elif "cluster_sizes" in data:
            cs = np.asarray(data["cluster_sizes"], dtype=np.int64)
        else:
            raise KeyError(f"{pkl_path} missing cluster_size(s). Keys: {list(data.keys())}")

        if fps.shape[0] != cs.shape[0]:
            raise ValueError(f"{lib}: fingerprints rows ({fps.shape[0]}) != cluster_sizes rows ({cs.shape[0]})")

        # offsets
        start = cursor
        end = cursor + fps.shape[0]
        lib_offsets[lib] = (start, end)
        cursor = end

        all_fps_packed.append(fps)
        all_cluster_sizes.append(cs)
        all_labels.extend([lib] * fps.shape[0])

        # Export data for selected decoys: read from decoy PKL if present; fill None otherwise
        if lib == DECOY_LIB and ("titles_medoids" in data) and ("smiles_medoids" in data):
            titles = np.asarray(data["titles_medoids"], dtype=object)
            smiles = np.asarray(data["smiles_medoids"], dtype=object)
            if titles.shape[0] != fps.shape[0] or smiles.shape[0] != fps.shape[0]:
                raise ValueError(f"{lib}: titles/smiles length mismatch with fingerprints")
            all_titles.extend(titles.tolist())
            all_smiles.extend(smiles.tolist())
        else:
            all_titles.extend([None] * fps.shape[0])
            all_smiles.extend([None] * fps.shape[0])

    if not all_fps_packed:
        raise RuntimeError(f"No libraries loaded from {LIBRARIES_DIR}")

    fps_packed = np.concatenate(all_fps_packed, axis=0)
    cluster_sizes_all = np.concatenate(all_cluster_sizes, axis=0)
    labels_all = np.asarray(all_labels, dtype=object)
    all_titles = np.asarray(all_titles, dtype=object)
    all_smiles = np.asarray(all_smiles, dtype=object)

    decoy_mask = (labels_all == DECOY_LIB)
    n_decoys = int(decoy_mask.sum())

    print("\nLoaded libraries:", loaded_libs)
    print("Total medoids (clustered):", int(fps_packed.shape[0]))
    print("Decoy medoids inside clustered set:", n_decoys)

    if not exclude_decoys_from_clustering:
        if n_decoys <= 0:
            raise RuntimeError(f"No decoy medoids found for {DECOY_LIB}. Check folder name and PKL presence.")

        # K is defined by #active medoids, even if actives are excluded from clustering
        K = 1 * read_active_K()
        print("Target K (decoys selected) = #actives:", K)

        if K <= 0:
            raise RuntimeError("K computed as 0; check actives PKL.")
        if n_decoys < K:
            raise RuntimeError(f"Not enough decoy medoids ({n_decoys}) to match actives ({K}).")
    else:
        K = None
        print("[INFO] Decoys are excluded from re-clustering; decoy selection stage is skipped in this run.")

    # ----------------------------
    # Estimate BitBirch threshold (packed)
    # ----------------------------
    rep = iSIM.jt_stratified_sampling(fps_packed, n_samples=6500)
    fps_rep_packed = fps_packed[rep]  # already packed
    sim = iSIM.jt_sim_matrix_packed(fps_rep_packed)
    sim = sim[~np.eye(sim.shape[0], dtype=bool)]

    avg = float(sim.mean())
    std = float(sim.std())
    threshold = avg + THRESH_Z * std
    extra_thr = std if RECLUSTER_EXTRA is None else float(RECLUSTER_EXTRA)

    print("\nGlobal BitBirch threshold:", threshold)
    print("Similarity mean/std:", avg, std)
    print("Recluster extra_threshold:", extra_thr)

    # ----------------------------
    # Global BitBirch clustering (packed)
    # ----------------------------
    start = time.time()
    bb = BitBirch(
        branching_factor=BRANCHING_FACTOR,
        threshold=threshold,
        merge_criterion=MERGE_CRITERION,
    )
    bb.fit(fps_packed)
    bb.recluster_inplace(
        iterations=RECLUSTER_ITERS,
        extra_threshold=extra_thr,
        shuffle=recluster_shuffle,
        verbose=True,
    )
    clusters = bb.get_cluster_mol_ids()
    print(f"\nProduced {len(clusters)} global clusters in {time.time()-start:.2f}s")

    # ----------------------------
    # Build global cluster summary
    #   - decoy mass: for reporting/coverage
    #   - vs mass: for allocation + spearman (exclude decoys and actives)
    # ----------------------------
    global_clusters = []
    global_cluster_decoy_mass = np.zeros(len(clusters), dtype=np.int64)
    global_cluster_decoy_count = np.zeros(len(clusters), dtype=np.int64)
    global_cluster_active_mass = np.zeros(len(clusters), dtype=np.int64)
    global_cluster_active_count = np.zeros(len(clusters), dtype=np.int64)
    global_cluster_vs_mass = np.zeros(len(clusters), dtype=np.int64)

    for gid, cluster in enumerate(clusters):
        members = np.asarray(cluster, dtype=np.int64)

        # decoy-only mass
        dec_members = members[labels_all[members] == DECOY_LIB]
        dm = int(cluster_sizes_all[dec_members].sum()) if dec_members.size else 0
        global_cluster_decoy_mass[gid] = dm
        global_cluster_decoy_count[gid] = int(dec_members.size)

        act_members = members[labels_all[members] == ACTIVE_LIB]
        am = int(cluster_sizes_all[act_members].sum()) if act_members.size else 0
        global_cluster_active_mass[gid] = am
        global_cluster_active_count[gid] = int(act_members.size)

        # VS-only mass (exclude decoys + actives)
        is_vs = (labels_all[members] != DECOY_LIB) & (labels_all[members] != ACTIVE_LIB)
        vs_members = members[is_vs]
        vm = int(cluster_sizes_all[vs_members].sum()) if vs_members.size else 0
        global_cluster_vs_mass[gid] = vm

        rep_mass = int(cluster_sizes_all[members].sum()) if members.size else 0
        lib_mass = {}
        for idx in members:
            lib = labels_all[idx]
            lib_mass[lib] = lib_mass.get(lib, 0) + int(cluster_sizes_all[idx])

        global_clusters.append({
            "global_cluster_id": int(gid),
            "medoid_indices": members.astype(np.int64),
            "represented_mass_all": int(rep_mass),
            "decoy_mass": int(dm),
            "active_mass": int(am),
            "vs_mass": int(vm),
            "library_mass": lib_mass,
            "n_medoids": int(members.size),
            "n_active_medoids": int(act_members.size),
        })

    sum_vs_mass = float(global_cluster_vs_mass.sum())
    if sum_vs_mass <= 0:
        raise RuntimeError("VS mass sum across global clusters is 0; cannot allocate. "
                           "Did you load any vendor libraries besides decoys?")

    if exclude_decoys_from_clustering:
        # In this mode we only export inter-library clusters (including singleton clusters).
        with open(OUT_PKL, "wb") as f:
            pickle.dump(
                {
                    "global_clusters": global_clusters,
                    "labels": labels_all,
                    "cluster_sizes": cluster_sizes_all,
                    "loaded_libraries": loaded_libs,
                    "lib_offsets": lib_offsets,
                    "bitbirch": {
                        "threshold": float(threshold),
                        "sim_mean": float(avg),
                        "sim_std": float(std),
                        "extra_threshold": float(extra_thr),
                        "branching_factor": int(BRANCHING_FACTOR),
                        "merge_criterion": str(MERGE_CRITERION),
                        "recluster_iters": int(RECLUSTER_ITERS),
                        "thresh_z": float(THRESH_Z),
                        "n_bits": int(N_BITS),
                    },
                    "selection": {
                        "mode": "cluster_only_decoys_excluded",
                        "ACTIVE_LIB": ACTIVE_LIB,
                        "DECOY_LIB": DECOY_LIB,
                    },
                },
                f,
            )

        np.save(OUT_NPY, np.array([], dtype=np.int64))

        with open(OUT_TXT, "w") as f:
            f.write("mode	cluster_only_decoys_excluded\n")
            f.write(f"n_global_clusters	{int(len(global_clusters))}\n")
            f.write(f"sum_vs_mass	{int(sum_vs_mass)}\n")

        print(f"\nSaved: {OUT_PKL}")
        print(f"Saved: {OUT_NPY}")
        print(f"Saved: {OUT_TXT}")
        return

# ----------------------------
# Allocate COUNT quotas per global cluster
#   Approach 2: exact top-K, one decoy per cluster.
#   Approach 3: anchor + weighted proportional allocation with deterministic integer rounding.
# ----------------------------

    analog_target = int(round(float(K) * analog_fraction))
    analog_selected, analog_counts = select_active_cluster_analogs(
        global_clusters,
        labels_all,
        cluster_sizes_all,
        ACTIVE_LIB,
        DECOY_LIB,
        analog_target,
    )
    if analog_selected.size < analog_target:
        print(
            f"[WARN] Requested {analog_target} active-adjacent analog decoys but only "
            f"{analog_selected.size} were available."
        )

    K_main = K - int(analog_selected.size)
    remaining_decoy_count = np.maximum(global_cluster_decoy_count - analog_counts, 0)

    eligible = np.where(remaining_decoy_count > 0)[0]
    if K_main > 0 and eligible.size < K_main:
        raise RuntimeError(
            f"Not enough decoy-containing clusters for allocation: eligible={eligible.size}, K_main={K_main}"
        )

    eligible_vs = global_cluster_vs_mass[eligible].astype(np.float64)
    order = np.argsort(-eligible_vs, kind="mergesort")
    ranked = eligible[order]

    q_cnt = np.zeros(len(global_clusters), dtype=np.int64)
    expected_alloc = np.zeros(len(global_clusters), dtype=np.float64)

    if K_main <= 0:
        chosen = np.array([], dtype=np.int64)
    elif approach == 2:
        chosen = ranked[:K_main]
        q_cnt[chosen] = 1
        expected_alloc[chosen] = 1.0
        print("\n[Approach 2] Top-K one-per-cluster allocation")

    elif approach == 3:
        # 1) Deterministic anchor: one decoy from top-N clusters by VS mass
        anchor_n = min(approach3_anchor_topk, K_main, int(ranked.size))
        anchors = ranked[:anchor_n]
        q_cnt[anchors] = 1
        expected_alloc[anchors] = 1.0

        remaining = K_main - int(anchor_n)

        # 2) Weighted proportional allocation over top-M pool (can include anchors)
        pool_n = min(max(anchor_n, approach3_pool_topk), int(ranked.size))
        pool = ranked[:pool_n]

        if remaining > 0 and pool.size > 0:
            w = global_cluster_vs_mass[pool].astype(np.float64)
            if approach3_weight_exp != 1.0:
                w = np.power(w, approach3_weight_exp)

            ws = float(w.sum())
            if ws <= 0:
                w = np.ones(pool.size, dtype=np.float64)
                ws = float(w.sum())

            p = w / ws
            theo_extra = float(remaining) * p
            expected_alloc[pool] += theo_extra

            cap_pool = remaining_decoy_count[pool].astype(np.int64) - q_cnt[pool]
            cap_pool = np.maximum(cap_pool, 0)

            q_extra = allocate_counts_with_capacity(theo_extra, cap_pool, remaining)
            q_cnt[pool] += q_extra

        if int(q_cnt.sum()) != K_main:
            raise RuntimeError(
                f"Approach 3 allocation failed: sum(q_cnt)={int(q_cnt.sum())} != K_main={K_main}"
            )

        print("\n[Approach 3] Anchor + weighted proportional allocation")
        print(
            f"anchor_topk={anchor_n} | pool_topk={pool_n} | "
            f"weight_exp={approach3_weight_exp:.4f}"
        )

    else:
        raise RuntimeError(f"Unsupported decoy selection approach: {approach}")

    chosen = np.where(q_cnt > 0)[0]
    print("Eligible (decoy-containing) clusters:", int(eligible.size))
    print("Chosen clusters:", int(chosen.size))
    print("Analog decoys reserved:", int(analog_selected.size))
    print("Sum(q_cnt):", int(q_cnt.sum()))
    if chosen.size:
        print("Chosen vs_mass percentiles [0,25,50,75,90,95,99,100]:")
        print(np.percentile(global_cluster_vs_mass[chosen], [0,25,50,75,90,95,99,100]))

    # ----------------------------
    # Select exactly K decoy medoids, cluster-by-cluster
    # ----------------------------
    selected = []

    for gid, gc in enumerate(global_clusters):
        need = int(q_cnt[gid])
        if need <= 0:
            continue

        members = gc["medoid_indices"]
        dec_members = members[labels_all[members] == DECOY_LIB]
        if analog_selected.size:
            dec_members = dec_members[~np.isin(dec_members, analog_selected)]
        if dec_members.size == 0:
            continue

        masses = cluster_sizes_all[dec_members].astype(np.int64)

        order = np.argsort(-masses, kind="mergesort")  # deterministic largest-mass-first within each cluster

        take = dec_members[order[:min(need, dec_members.size)]]
        if take.size:
            selected.append(take)

    selected_main = np.concatenate(selected) if selected else np.array([], dtype=np.int64)
    selected_medoids = np.concatenate([analog_selected, selected_main]) if analog_selected.size else selected_main

    if selected_medoids.size < K:
        selected_set = set(map(int, selected_medoids.tolist()))
        remaining = np.where(decoy_mask)[0]
        remaining = remaining[~np.isin(remaining, np.fromiter(selected_set, dtype=np.int64))]
        rem_order = np.argsort(-cluster_sizes_all[remaining], kind="mergesort")
        add = remaining[rem_order[: (K - selected_medoids.size)]]
        selected_medoids = np.concatenate([selected_medoids, add])

    if selected_medoids.size != K:
        raise RuntimeError(f"Selection size {selected_medoids.size} != K {K}")
    # ----------------------------
    # Metrics
    # ----------------------------
    selected_count_per_cluster = np.zeros(len(global_clusters), dtype=np.int64)
    touched = 0

    n_clusters = len(global_clusters)
    vs_mass = np.zeros(n_clusters, dtype=np.int64)         # background cluster size (exclude decoys+actives)
    n_decoy_medoids = np.zeros(n_clusters, dtype=np.int64) # available decoy medoids per cluster (B1)

    for gid, gc in enumerate(global_clusters):
        members = gc["medoid_indices"]

        # Selected count in this global cluster
        cnt = int(np.sum(np.isin(members, selected_medoids)))
        selected_count_per_cluster[gid] = cnt
        if cnt > 0:
            touched += 1

        # VS mass (exclude decoys and actives)
        vs_members = members[
            (labels_all[members] != DECOY_LIB) &
            (labels_all[members] != ACTIVE_LIB)
        ]
        vs_mass[gid] = int(cluster_sizes_all[vs_members].sum())

        # Available decoys (B1): number of decoy medoids in this global cluster
        dec_members = members[labels_all[members] == DECOY_LIB]
        n_decoy_medoids[gid] = int(dec_members.size)

    analog_cluster_mask = (global_cluster_active_count > 0) & (global_cluster_decoy_count > 0)
    main_selected_per_cluster = np.maximum(selected_count_per_cluster - analog_counts, 0)
    overlap_cluster_mask = analog_cluster_mask & (main_selected_per_cluster > 0)

    active_cluster_diag = pd.DataFrame({
        "cluster_id": np.arange(len(global_clusters), dtype=int),
        "represented_mass_all": np.array([int(gc["represented_mass_all"]) for gc in global_clusters], dtype=int),
        "vs_mass": global_cluster_vs_mass.astype(int),
        "active_medoids": global_cluster_active_count.astype(int),
        "decoy_medoids": global_cluster_decoy_count.astype(int),
        "analog_reserved_decoys": analog_counts.astype(int),
        "main_selected_decoys": main_selected_per_cluster.astype(int),
        "total_selected_decoys": selected_count_per_cluster.astype(int),
        "active_cluster": analog_cluster_mask.astype(int),
    })
    active_cluster_diag = active_cluster_diag.loc[active_cluster_diag["active_cluster"] == 1].copy()
    active_cluster_diag = active_cluster_diag.sort_values(
        ["represented_mass_all", "active_medoids", "decoy_medoids"],
        ascending=[False, False, False],
    )
    active_cluster_diag.to_csv(OUT_ACTIVE_CLUSTER_OVERLAP_CSV, index=False)

    print("\n=== Active-Cluster Overlap Diagnostics ===")
    print("Available clusters with actives and decoys:", int(np.sum(analog_cluster_mask)))
    print("Total decoy medoids in active-containing clusters:", int(global_cluster_decoy_count[analog_cluster_mask].sum()))
    print("Analog decoys reserved from active-containing clusters:", int(analog_counts.sum()))
    print("Main-strategy decoys taken from active-containing clusters:", int(main_selected_per_cluster[analog_cluster_mask].sum()))
    print("Active-containing clusters also touched by main strategy:", int(np.sum(overlap_cluster_mask)))
    print(f"Wrote active-cluster diagnostics CSV: {OUT_ACTIVE_CLUSTER_OVERLAP_CSV} ({len(active_cluster_diag)} rows)")
    if not active_cluster_diag.empty:
        print(active_cluster_diag.head(20).to_string(index=False))

    # Denominator: clusters that contain at least one decoy medoid (B1 notion of "non-empty decoy clusters")
    nonempty_decoy_clusters = int(np.sum(n_decoy_medoids > 0))

    touched_ratio = touched / nonempty_decoy_clusters if nonempty_decoy_clusters > 0 else np.nan
    selected_mass_total = int(cluster_sizes_all[selected_medoids].sum())

    print("\nSelected decoy medoids:", int(selected_medoids.size))
    print("Selected represented mass total:", selected_mass_total)
    print(
        f"Touched decoy-containing global clusters: "
        f"{touched}/{nonempty_decoy_clusters} = {touched_ratio:.4f}"
    )


    mc_decoys = mass_coverage_topk_global(
        selected_medoids,
        cluster_sizes_all,
        eligible_mask=decoy_mask,
        K=K,
    )
    print("Internal mass coverage vs top-K decoy medoids:", mc_decoys)

    # Only clusters that actually contain decoys
    mask = (vs_mass > 0) & (n_decoy_medoids > 0)
    rho = spearman_rho(vs_mass[mask], selected_count_per_cluster[mask])
    print(f"Spearman rho (VS mass vs selected decoys per cluster): {rho:.4f}")
    # ----------------------------
    # VS-mass coverage over pickable clusters (>1 decoy medoid)
    #   - cluster mass := VS-only mass (exclude decoys + actives)
    #   - numerator := sum VS mass over clusters that were touched by selected decoys
    #   - denominator := sum VS mass over all pickable clusters (n_decoy_medoids > 1)
    # ----------------------------

    pickable_mask = n_decoy_medoids > 0              # "pickable clusters"
    touched_mask = selected_count_per_cluster > 0    # clusters where you selected >=1 decoy

    # Restrict numerator to touched clusters that are pickable (should usually be true, but safe)
    num_mask = touched_mask & pickable_mask

    picked_vs_mass = int(vs_mass[num_mask].sum())
    total_vs_mass_pickable = int(vs_mass[pickable_mask].sum())

    vs_mass_coverage_pickable = (
        picked_vs_mass / total_vs_mass_pickable
        if total_vs_mass_pickable > 0 else np.nan
    )

    print("\n=== VS-mass coverage (pickable clusters only) ===")
    print("Pickable clusters (n_decoy_medoids > 1):", int(np.sum(pickable_mask)))
    print("Touched pickable clusters:", int(np.sum(num_mask)))
    print("Picked VS mass (sum over touched pickable clusters):", picked_vs_mass)
    print("Total VS mass (sum over all pickable clusters):", total_vs_mass_pickable)
    print("VS-mass coverage over pickable clusters:", vs_mass_coverage_pickable)


    if approach == 3:
        represented_mass_all = np.array([int(gc.get("represented_mass_all", 0)) for gc in global_clusters], dtype=np.int64)
        cluster_n_medoids = np.array([int(gc.get("n_medoids", 0)) for gc in global_clusters], dtype=np.int64)

        alloc_df = pd.DataFrame({
            "cluster_id": np.arange(len(global_clusters), dtype=int),
            "represented_vs_mass": vs_mass.astype(int),
            "cluster_size_all": represented_mass_all.astype(int),
            "cluster_n_medoids": cluster_n_medoids.astype(int),
            "available_decoys": n_decoy_medoids.astype(int),
            "theoretical_allocated": expected_alloc.astype(float),
            "planned_allocated": q_cnt.astype(int),
            "allocated_decoys": selected_count_per_cluster.astype(int),
        })

        # helpful diagnostics columns
        alloc_df["realization_gap"] = alloc_df["allocated_decoys"] - alloc_df["theoretical_allocated"]
        alloc_df["plan_vs_actual_gap"] = alloc_df["allocated_decoys"] - alloc_df["planned_allocated"]

        alloc_df = alloc_df.sort_values("represented_vs_mass", ascending=False, kind="mergesort").reset_index(drop=True)
        alloc_df.insert(1, "rank", np.arange(1, len(alloc_df) + 1, dtype=int))

        alloc_df.to_csv(OUT_APPROACH3_ALLOC_CSV, index=False)
        print(f"Wrote approach-3 allocation CSV: {OUT_APPROACH3_ALLOC_CSV} ({len(alloc_df)} rows)")

    # ----------------------------
    # Export selected decoys benchmark CSV (Title, SMILES)
    # ----------------------------
    sel_dec = selected_medoids[labels_all[selected_medoids] == DECOY_LIB]
    if sel_dec.size != K:
        # This should not happen, but keep a clear failure mode
        raise RuntimeError(f"Expected K decoy selections, got {sel_dec.size}. Check label mapping.")

    #x = vs_mass[mask].astype(np.float64)
    #quota = q_cnt_float[mask].astype(np.float64)   # or use q_cnt[mask] for integer quota
    #sel = selected_count_per_cluster[mask].astype(np.float64)
    #avail = n_decoy_medoids[mask].astype(np.float64)


    # If your decoy PKL has titles_medoids/smiles_medoids, these are filled; otherwise they are None.
    titles = all_titles[sel_dec]
    smiles = all_smiles[sel_dec]
    if np.any(pd.isna(smiles)) or np.any(pd.isna(titles)):
        raise RuntimeError(
            "Missing Title/SMILES for some selected decoys. "
            "Ensure UF-Scripps-Decoys/npy_medoids.pkl contains 'titles_medoids' and 'smiles_medoids'."
        )

    out_df = pd.DataFrame({
        "Title": titles.astype(str),
        "SMILES": smiles.astype(str),
        "global_medoid_index": sel_dec.astype(int),
        "represented_mass": cluster_sizes_all[sel_dec].astype(int),
    }).sort_values("represented_mass", ascending=False, kind="mergesort").reset_index(drop=True)

    out_df.to_csv(OUT_SELECTED_CSV, index=False)
    print(f"\nWrote benchmarking dataset CSV: {OUT_SELECTED_CSV} ({len(out_df)} rows)")

    # ----------------------------
    # Save outputs
    # ----------------------------
    with open(OUT_PKL, "wb") as f:
        pickle.dump(
            {
                "global_clusters": global_clusters,
                "labels": labels_all,
                "cluster_sizes": cluster_sizes_all,
                "loaded_libraries": loaded_libs,
                "lib_offsets": lib_offsets,

                "bitbirch": {
                    "threshold": float(threshold),
                    "sim_mean": float(avg),
                    "sim_std": float(std),
                    "extra_threshold": float(extra_thr),
                    "branching_factor": int(BRANCHING_FACTOR),
                    "merge_criterion": str(MERGE_CRITERION),
                    "recluster_iters": int(RECLUSTER_ITERS),
                    "thresh_z": float(THRESH_Z),
                    "n_bits": int(N_BITS),
                },

                "selection": {
                    "ACTIVE_LIB": ACTIVE_LIB,
                    "DECOY_LIB": DECOY_LIB,
                    "K": int(K),
                    "WEIGHTED_ORDER_WITHIN_CLUSTER": bool(WEIGHTED_ORDER_WITHIN_CLUSTER),
                    "RNG_SEED": int(rng_seed),
                    "RECLUSTER_SHUFFLE": bool(recluster_shuffle),
                    "approach": int(approach),
                    "approach3_anchor_topk": int(approach3_anchor_topk),
                    "approach3_pool_topk": int(approach3_pool_topk),
                    "approach3_weight_exp": float(approach3_weight_exp),
                    "analog_fraction": float(analog_fraction),
                    "analog_selected_count": int(analog_selected.size),

                    #"targets_cnt": targets_cnt,
                    "q_cnt": q_cnt,
                    #"cap_stats_cnt": cap_stats,

                    "selected_medoids_indices": selected_medoids,
                    "selected_mass_total": int(selected_mass_total),

                    "touched_clusters": int(touched),
                    "n_global_clusters": int(len(global_clusters)),
                    "touched_ratio": float(touched_ratio),

                    "mass_coverage_topk_decoys": float(mc_decoys) if np.isfinite(mc_decoys) else mc_decoys,
                    "spearman_rho_vs_mass_vs_selected_count": float(rho) if np.isfinite(rho) else rho,
                },
            },
            f,
        )

    np.save(OUT_NPY, selected_medoids)

    with open(OUT_TXT, "w") as f:
        f.write(f"K_selected_decoys\t{int(selected_medoids.size)}\n")
        f.write(f"selected_mass_total\t{int(selected_mass_total)}\n")
        f.write(f"touched_clusters\t{int(touched)}\n")
        f.write(f"n_global_clusters\t{int(len(global_clusters))}\n")
        f.write(f"touched_ratio\t{float(touched_ratio)}\n")
        f.write(f"mass_coverage_topk_decoys\t{float(mc_decoys)}\n")
        f.write(f"spearman_rho_vs_mass_vs_selected_count\t{float(rho)}\n")
        f.write(f"picked_vs_mass_touched_pickable\t{int(picked_vs_mass)}\n")
        f.write(f"total_vs_mass_all_pickable\t{int(total_vs_mass_pickable)}\n")
        f.write(f"vs_mass_coverage_pickable\t{float(vs_mass_coverage_pickable)}\n")

    print(f"\nSaved: {OUT_PKL}")
    print(f"Saved: {OUT_NPY}")
    print(f"Saved: {OUT_TXT}")


if __name__ == "__main__":
    main()
