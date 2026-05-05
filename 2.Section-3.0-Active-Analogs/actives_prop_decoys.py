#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import bblean
import bblean.similarity as iSIM
from bblean.bitbirch import BitBirch
from bblean.fingerprints import pack_fingerprints
import typing as tp
from numpy.typing import NDArray, DTypeLike
from rdkit.Chem import MolFromSmiles, SanitizeMol
from rdkit.Chem import rdFingerprintGenerator
from matplotlib_venn import venn2

def pick_col(df, candidates, explicit=None, required=True):
    if explicit:
        if explicit not in df.columns:
            raise RuntimeError(f"Requested column '{explicit}' not found")
        return explicit
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise RuntimeError(f"Missing required column. Tried: {candidates}")
    return None
def plot_tanimoto_histogram(analog_pool, actives, out_png, radius=2, n_bits=2048):
    """
    Histogram of maximum Tanimoto similarity of mixed-cluster decoys to any active.
    """
    if analog_pool.empty:
        print("[WARN] No decoys in mixed clusters for histogram")
        return
    
    # Calculate max Tanimoto for ALL decoys in mixed clusters (not just selected)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    
    # Generate fingerprints once
    act_fps = [generator.GetFingerprint(Chem.MolFromSmiles(smi)) 
              for smi in actives["_canon_smiles"] if Chem.MolFromSmiles(smi)]
    
    dec_fps = []
    tanimoto_max = []
    
    for smi in analog_pool["_canon_smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = generator.GetFingerprint(mol)
            sims = DataStructs.BulkTanimotoSimilarity(fp, act_fps)
            tanimoto_max.append(float(max(sims)) if sims else 0.0)
            dec_fps.append(fp)
    
    if not tanimoto_max:
        print("[WARN] No valid fingerprints for histogram")
        return
    
    tanimoto_max = np.array(tanimoto_max)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), height_ratios=[3, 1])
    
    # Main histogram
    bins = np.histogram_bin_edges(tanimoto_max, bins='auto')
    n, bins, patches = ax1.hist(tanimoto_max, bins=bins, alpha=0.7, 
                               color='#d95f02', edgecolor='black', linewidth=0.5,
                               density=True, label=f'N={len(tanimoto_max):,}')
    
    # Styling
    ax1.axvline(tanimoto_max.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {tanimoto_max.mean():.3f}')
    ax1.axvline(np.median(tanimoto_max), color='orange', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(tanimoto_max):.3f}')
    ax1.axvline(tanimoto_max.max(), color='green', linestyle='-', linewidth=2, 
                label=f'Max: {tanimoto_max.max():.3f}')
    
    ax1.set_xlabel('Maximum Tanimoto Similarity to Any Active')
    ax1.set_ylabel('Density')
    ax1.set_title('Tanimoto Similarity Distribution: Decoys in Mixed Clusters', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Summary stats box
    stats_text = (
        f"N decoys: {len(tanimoto_max):,}\n"
        f"Mean: {tanimoto_max.mean():.3f}\n"
        f"Median: {np.median(tanimoto_max):.3f}\n"
        f"Std: {tanimoto_max.std():.3f}\n"
        f"25th %ile: {np.percentile(tanimoto_max, 25):.3f}\n"
        f"75th %ile: {np.percentile(tanimoto_max, 75):.3f}\n"
        f"Max: {tanimoto_max.max():.3f}"
    )
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                      edgecolor='gray', alpha=0.9, linewidth=1))
    
    # Cumulative distribution (bottom subplot)
    sorted_sims = np.sort(tanimoto_max)
    cdf = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    ax2.plot(sorted_sims, cdf, color='#d95f02', linewidth=2)
    ax2.set_xlabel('Maximum Tanimoto Similarity')
    ax2.set_ylabel('Cumulative\nFraction')
    ax2.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Threshold lines on CDF
    ax2.axvline(tanimoto_max.mean(), color='red', linestyle='--', alpha=0.7, label='Mean')
    ax2.axvline(np.percentile(tanimoto_max, 75), color='orange', linestyle='--', alpha=0.7, label='75th %ile')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Tanimoto histogram saved: {out_png}")
    print(f"   Range: {tanimoto_max.min():.3f} - {tanimoto_max.max():.3f}")
    print(f"   Mean/Median: {tanimoto_max.mean():.3f} / {np.median(tanimoto_max):.3f}")

def select_analog_decoys(actives_df, decoys_df, analog_target, radius, n_bits):
    """
    Select mixed-cluster decoys by prioritizing clusters with more actives.

    Clusters are ordered by `active_count` descending, then `decoy_count`
    descending. Within each cluster, decoys are ranked by Tanimoto similarity to
    that cluster's active medoid. Selection proceeds round-robin across the
    ordered clusters until the requested mixed-decoy quota is filled.
    """
    if analog_target <= 0 or decoys_df.empty:
        return np.array([], dtype=int), np.array([], dtype=float)

    required_cols = {"_cluster_id", "active_count", "decoy_count"}
    missing_decoy_cols = required_cols.difference(decoys_df.columns)
    if missing_decoy_cols:
        raise RuntimeError(
            f"Missing required cluster columns for mixed-decoy selection: {sorted(missing_decoy_cols)}"
        )
    if "_cluster_id" not in actives_df.columns:
        raise RuntimeError("Actives dataframe must include '_cluster_id' to compute cluster medoids")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    active_fp_by_cluster = {}
    for cluster_id, sub_df in actives_df.groupby("_cluster_id"):
        cluster_fps = []
        for smi in sub_df["_canon_smiles"]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                cluster_fps.append(generator.GetFingerprint(mol))
        if not cluster_fps:
            continue
        if len(cluster_fps) == 1:
            active_fp_by_cluster[int(cluster_id)] = cluster_fps[0]
            continue

        mean_sims = []
        for i, fp in enumerate(cluster_fps):
            others = cluster_fps[:i] + cluster_fps[i + 1:]
            sims = DataStructs.BulkTanimotoSimilarity(fp, others)
            mean_sims.append(float(np.mean(sims)) if len(sims) else 1.0)
        active_fp_by_cluster[int(cluster_id)] = cluster_fps[int(np.argmax(mean_sims))]

    ranked = decoys_df.copy().reset_index(drop=True)
    medoid_scores = []
    valid_pool_indices = []

    for pool_idx, smi in enumerate(ranked["_canon_smiles"]):
        cluster_id = int(ranked.iloc[pool_idx]["_cluster_id"])
        medoid_fp = active_fp_by_cluster.get(cluster_id)
        mol = Chem.MolFromSmiles(smi)
        if medoid_fp is None or mol is None:
            medoid_scores.append(np.nan)
            continue
        dec_fp = generator.GetFingerprint(mol)
        medoid_scores.append(float(DataStructs.TanimotoSimilarity(dec_fp, medoid_fp)))
        valid_pool_indices.append(pool_idx)

    ranked["_nearest_score"] = np.asarray(medoid_scores, dtype=float)
    ranked = ranked.dropna(subset=["_nearest_score"]).copy()
    if ranked.empty:
        return np.array([], dtype=int), np.array([], dtype=float)

    cluster_order_df = (
        ranked[["_cluster_id", "active_count", "decoy_count"]]
        .drop_duplicates()
        .sort_values(
            ["active_count", "decoy_count", "_cluster_id"],
            ascending=[False, False, True],
            kind="mergesort",
        )
    )
    cluster_order = cluster_order_df["_cluster_id"].tolist()

    ranked = ranked.reset_index().rename(columns={"index": "_pool_index"})
    per_cluster = {}
    for cluster_id in cluster_order:
        cluster_rows = ranked.loc[ranked["_cluster_id"] == cluster_id].sort_values(
            ["_nearest_score", "_pool_index"],
            ascending=[False, True],
            kind="mergesort",
        )
        per_cluster[int(cluster_id)] = cluster_rows["_pool_index"].tolist()

    chosen = []
    cluster_ptr = {int(cluster_id): 0 for cluster_id in cluster_order}
    target = min(int(analog_target), len(ranked))

    while len(chosen) < target:
        picked_this_round = False
        for cluster_id in cluster_order:
            cluster_id = int(cluster_id)
            ptr = cluster_ptr[cluster_id]
            members = per_cluster[cluster_id]
            if ptr >= len(members):
                continue
            chosen.append(int(members[ptr]))
            cluster_ptr[cluster_id] = ptr + 1
            picked_this_round = True
            if len(chosen) >= target:
                break
        if not picked_this_round:
            break

    return np.asarray(chosen, dtype=int), ranked["_nearest_score"].to_numpy(dtype=float)

def canonicalize_smiles(val):
    if pd.isna(val):
        return None
    mol = Chem.MolFromSmiles(str(val))
    if mol is None:
        return None       
    return Chem.MolToSmiles(mol, canonical=True)


def prepare_dataframe(df, smiles_col=None, title_col=None, label_value=None):
    smiles_col = pick_col(df, ["SMILES", "Smiles", "smiles"], explicit=smiles_col)
    title_col = pick_col(df, ["Title", "title", "Sample_ID", "ID", "id", "Name"], explicit=title_col, required=False)

    out = pd.DataFrame()
    out["SMILES"] = df[smiles_col].astype(str)
    out["Title"] = df[title_col].astype(str) if title_col else [f"row_{i}" for i in range(len(df))]
    out["Label"] = int(label_value) if label_value is not None else pd.to_numeric(
        df[pick_col(df, ["Label", "Label ", "label", "Actividad"])], errors="coerce"
    ).astype("Int64")
    out["_canon_smiles"] = out["SMILES"].map(canonicalize_smiles)

    invalid = int(out["_canon_smiles"].isna().sum())
    if invalid:
        print(f"[WARN] Invalid SMILES removed: {invalid}")
    out = out.dropna(subset=["_canon_smiles"]).copy()
    out = out.drop_duplicates(subset="_canon_smiles", keep="first").reset_index(drop=True)
    return out

from matplotlib_venn import venn2, venn2_circles
import matplotlib.pyplot as plt

def plot_cluster_venn(cluster_stats, out_png):
    """
    Cleaner cluster-composition summary plot.

    A Venn-style diagram becomes unreadable when pure decoy clusters dominate,
    so this plot shows the three cluster designations as annotated bars.
    """
    n_actives = int(cluster_stats["active_count"].sum())
    n_decoys = int(cluster_stats["decoy_count"].sum())
    n_clusters = int(len(cluster_stats))

    n_pure_actives = int(((cluster_stats["active_count"] > 0) & (cluster_stats["decoy_count"] == 0)).sum())
    n_pure_decoys = int(((cluster_stats["active_count"] == 0) & (cluster_stats["decoy_count"] > 0)).sum())
    n_mixed = int(((cluster_stats["active_count"] > 0) & (cluster_stats["decoy_count"] > 0)).sum())

    labels = ["Pure active", "Mixed", "Pure decoy"]
    values = [n_pure_actives, n_mixed, n_pure_decoys]
    colors = ["#1b9e77", "#7570b3", "#d95f02"]
    fractions = [value / n_clusters for value in values]

    fig, (ax_bar, ax_text) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [2.2, 1.0]}
    )

    bars = ax_bar.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2)
    ax_bar.set_title("BitBirch Cluster Composition", fontsize=16, fontweight="bold", pad=12)
    ax_bar.set_ylabel("Number of clusters", fontsize=12)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.35)
    ax_bar.set_axisbelow(True)

    ymax = max(values) * 1.08 if values else 1
    ax_bar.set_ylim(0, ymax)
    for bar, count, frac in zip(bars, values, fractions):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.01,
            f"{count:,}\n({frac:.1%})",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="semibold",
        )

    ax_text.axis("off")
    summary_text = (
        f"Total clusters: {n_clusters:,}\n\n"
        f"Pure active: {n_pure_actives:,}\n"
        f"Mixed: {n_mixed:,}\n"
        f"Pure decoy: {n_pure_decoys:,}\n\n"
        f"Total actives: {n_actives:,}\n"
        f"Total decoys: {n_decoys:,}\n\n"
        f"Mixed-cluster fraction: {n_mixed / n_clusters:.2%}"
    )
    ax_text.text(
        0.02,
        0.98,
        summary_text,
        transform=ax_text.transAxes,
        fontsize=12,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.95),
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def generate_dataset(actives, decoy_view, analog_pool, background_pool, target_decoys,
                    analog_fraction, rng, dataset_name):
    """Generate dataset with specified % decoys from mixed clusters."""
    analog_target = int(round(target_decoys * analog_fraction))

    analog_idx, nearest_scores = select_analog_decoys(
        actives, analog_pool, analog_target=analog_target,
        radius=2, n_bits=2048
    )
    analog_selected = analog_pool.iloc[analog_idx].copy()
    analog_selected["_fraction"] = f"{analog_fraction*100:.0f}%"
    analog_selected["_selection_source"] = "mixed"

    remaining_needed = target_decoys - len(analog_selected)
    random_idx = rng.choice(
        background_pool.index.to_numpy(),
        size=min(remaining_needed, len(background_pool)),
        replace=False,
    ) if remaining_needed > 0 else np.array([], dtype=int)
    random_selected = background_pool.loc[random_idx].copy()
    random_selected["_fraction"] = f"{analog_fraction*100:.0f}%"
    random_selected["_selection_source"] = "pure_decoy_random"

    selected_decoys = pd.concat([analog_selected, random_selected], ignore_index=True)
    benchmark = pd.concat([
        actives[["Title", "SMILES", "Label"]].copy(),
        selected_decoys[["Title", "SMILES", "Label"]].copy(),
    ], ignore_index=True)
    mixed_only_benchmark = pd.concat([
        actives[["Title", "SMILES", "Label"]].copy(),
        analog_selected[["Title", "SMILES", "Label"]].copy(),
    ], ignore_index=True)

    return benchmark, mixed_only_benchmark, len(analog_selected), len(random_selected)
def parse_args():
    p = argparse.ArgumentParser(description="BitBirch clustering with multiple decoy selection percentages")
    p.add_argument("--actives-csv", required=True)
    p.add_argument("--decoys-csv", required=True)
    p.add_argument("--actives-smiles-col", default=None)
    p.add_argument("--decoys-smiles-col", default=None)
    p.add_argument("--actives-title-col", default=None)
    p.add_argument("--decoys-title-col", default=None)
    p.add_argument("--ratio-k", type=float, default=1.0)
    p.add_argument("--branching-factor", type=int, default=100)
    p.add_argument("--merge-criterion", default="diameter")
    p.add_argument("--recluster-iters", type=int, default=8)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--clusters-out", default="bitbirch_clusters.csv")
    p.add_argument("--venn-plot", default="cluster_venn.png")
    p.add_argument("--active-cluster-plot", default="active_cluster_composition.png")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.random_state)

    actives = prepare_dataframe(
        pd.read_csv(args.actives_csv),
        smiles_col=args.actives_smiles_col,
        title_col=args.actives_title_col,
        label_value=1,
    )
    decoys = prepare_dataframe(
        pd.read_csv(args.decoys_csv),
        smiles_col=args.decoys_smiles_col,
        title_col=args.decoys_title_col,
        label_value=0,
    )

    if actives.empty:
        raise RuntimeError("No valid actives available after cleaning")
    if decoys.empty:
        raise RuntimeError("No valid decoys available after cleaning")

    target_decoys = int(round(len(actives) * float(args.ratio_k)))
    if target_decoys <= 0:
        raise RuntimeError("Target decoy count is 0; increase --ratio-k")
    if len(decoys) < target_decoys:
        raise RuntimeError(f"Decoy pool too small: need {target_decoys}, have {len(decoys)}")

    combined = pd.concat([actives, decoys], ignore_index=True)
    combined["_source"] = np.where(combined["Label"] == 1, "active", "decoy")
    smiles = combined["_canon_smiles"]
    fps = bblean.fps_from_smiles(smiles, pack=True, n_features=2048, kind="ecfp4")
    print(f"Shape: {fps.shape}, DType: {fps.dtype}")
    fps_unpacked = bblean.unpack_fingerprints(fps)
    print(f"Shape unpacked: {fps_unpacked.shape}, DType unpacked: {fps_unpacked.dtype}")
    fps_unpacked = bblean.unpack_fingerprints(fps)
    print(f"Shape unpacked: {fps_unpacked.shape}, DType unpacked: {fps_unpacked.dtype}")
    average_sim = iSIM.jt_isim_unpacked(fps_unpacked)
    print(f"Average similarity: {average_sim:.4f}")
    # Take a representative sample to estimate similarity std
    representative_samples = iSIM.jt_stratified_sampling(fps, n_samples=6000)

    # Calculate similarity matrix for the representative samples and exclude self-similarities
    sim_matrix = iSIM.jt_sim_matrix_packed(fps[representative_samples])
    sim_matrix = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]

    # Obtain mean and standard deviation
    _, std = np.mean(sim_matrix), np.std(sim_matrix)
    print(f"Estimated similarity mean: {average_sim:.4f}, std: {std:.4f}")
    optimal_threshold = average_sim + 5.5 * std
    bb_tree = bblean.BitBirch(branching_factor=1024, threshold=optimal_threshold, merge_criterion="diameter")

    # Cluster the packed fingerprints (By default all bblean functions take packed
    # fingerprints)
    bb_tree.fit(fps)
    clusters = bb_tree.get_cluster_mol_ids()
    print("Number of singletons", sum(1 for c in clusters if len(c) == 1))

    bb_tree.recluster_inplace(
        iterations=int(args.recluster_iters),
        extra_threshold=std,
        shuffle=True,
        verbose=True,
    )
    clusters = bb_tree.get_cluster_mol_ids()

    # Assign cluster IDs
    cluster_id = np.full(len(combined), -1, dtype=int)
    for gid, members in enumerate(clusters):
        cluster_id[np.asarray(members, dtype=int)] = int(gid)
    combined["_cluster_id"] = cluster_id

    # Cluster statistics
    cluster_stats = (combined.groupby("_cluster_id")["Label"]
                    .agg(active_count=lambda s: int(np.sum(s == 1)), 
                         decoy_count=lambda s: int(np.sum(s == 0)))
                    .reset_index())
    cluster_stats["active_cluster"] = cluster_stats["active_count"] > 0
    combined = combined.merge(cluster_stats, on="_cluster_id", how="left")

    # Save clusters and plots
    combined.to_csv(args.clusters_out, index=False)
    plot_cluster_venn(cluster_stats, args.venn_plot)

    # Prepare decoy pools
    active_view = combined.loc[combined["Label"] == 1].copy().reset_index(drop=True)
    decoy_view = combined.loc[combined["Label"] == 0].copy().reset_index(drop=True)
    analog_pool = decoy_view.loc[decoy_view["active_cluster"]].copy().reset_index(drop=True)
    background_pool = decoy_view.loc[~decoy_view["active_cluster"]].copy().reset_index(drop=True)

    print(f"Active-containing clusters: {len(analog_pool)} decoys")
    print(f"Pure decoy clusters: {len(background_pool)} decoys")
# Add this line after preparing analog_pool
    plot_tanimoto_histogram(analog_pool, actives, "mixed_decoys_tanimoto_histogram.png")
    # Generate 5 datasets (0%, 25%, 50%, 75%, 100% from mixed)
    fractions = [0.0, 0.25, 0.50, 0.75, 1.0]
    results = []
    
    for frac in fractions:
        dataset, mixed_only_dataset, n_analog, n_random = generate_dataset(
            active_view, decoy_view, analog_pool, background_pool, target_decoys,
            frac, rng, f"{int(frac*100)}pct_mixed"
        )
        results.append((frac, dataset, mixed_only_dataset, n_analog, n_random))

        out_name = f"benchmark_bitbirch_{int(frac*100)}pct_mixed.csv"
        mixed_only_name = f"benchmark_bitbirch_{int(frac*100)}pct_mixed_only.csv"
        dataset.to_csv(out_name, index=False)
        mixed_only_dataset.to_csv(mixed_only_name, index=False)
        print(f"Saved {out_name}: {len(dataset)} compounds "
              f"({n_analog} mixed + {n_random} pure decoy clusters)")
        print(f"Saved {mixed_only_name}: {len(mixed_only_dataset)} compounds "
              f"({n_analog} mixed decoys only)")

    # Summary stats
    print(f"\n=== SUMMARY ===")
    print(f"Total clusters: {len(cluster_stats)}")
    print(f"Active clusters: {int(cluster_stats['active_cluster'].sum())}")
    print(f"Actives: {len(actives)}")
    print(f"Target decoys per dataset: {target_decoys}")
    print(f"Generated datasets: {[f'{f*100:.0f}%' for f in fractions]}")
    print(f"Files: {args.clusters_out}, {args.venn_plot}, benchmarks (*.csv)")

if __name__ == "__main__":
    main()
