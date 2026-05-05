#!/usr/bin/env python3
import argparse
import os
import pickle
import random
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iSIM.comp import calculate_isim
from scipy.stats import spearmanr

from main_hexbin_functions import hexbin_density_autotune_nbins

# -----------------------------
# Paths / labels
# -----------------------------
NPZ_PATH_APPROACH_2 = "umap_approach_2_embedding.npz"
NPZ_PATH_APPROACH_3 = "umap_approach_3_embedding.npz"
DECOY_LIB  = "UF-Scripps-Decoys"
ACTIVE_LIB = "UF-Scripps-Actives"
LIBRARIES_DIR = "../../../../../Libraries"
PKL_NAME = "npy_medoids.pkl"


# -----------------------------
# Sweep settings
# -----------------------------
# -----------------------------
# Sweep settings
# -----------------------------
CAP_MAX_GRID = list(range(1, 21))   # run cap_max = 1..20

ALPHA = 1.0
CAP_REF_Q = 0.50


# -----------------------------
# Selection knobs (keep stable)
# -----------------------------
RATIO_K = 2

NBINS_MIN  = 15
NBINS_MAX  = 200
NBINS_STEP = 5

MIN_LOG_MASS = 0.0
RNG_SEED = 42
ISIM_GOAL = 0.10968

RDKit_OK = False
ISIM_OK = False

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import AllChem, DataStructs
    RDLogger.DisableLog("rdApp.warning")
    RDKit_OK = True
    

except Exception:
    RDKit_OK = False


def set_global_seed(seed):
    """
    Set deterministic RNG state used by numpy/python consumers in this script.
    """
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)



def load_active_count_from_pkl(libraries_dir=LIBRARIES_DIR, active_lib=ACTIVE_LIB, pkl_name=PKL_NAME):
    pkl_path = os.path.join(libraries_dir, active_lib, pkl_name)
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Missing actives PKL for ratio sizing: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    fps = np.asarray(data["fingerprints"], dtype=np.uint8)
    return int(fps.shape[0])


def inject_synthetic_actives_for_ratio(x, y, labels, titles, smiles, cluster_size, n_actives):
    """
    Keep actives excluded from structural workflow while preserving ratio_k*n_actives sizing
    expected by hexbin_density_autotune_nbins.
    """
    if n_actives <= 0:
        return x, y, labels, titles, smiles, cluster_size

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    labels = np.asarray(labels, dtype=object)
    titles = np.asarray(titles, dtype=object)
    smiles = np.asarray(smiles, dtype=object)
    cluster_size = np.asarray(cluster_size, dtype=np.int64)

    x_fill = np.full(n_actives, np.median(x), dtype=float)
    y_fill = np.full(n_actives, np.median(y), dtype=float)

    labels_fill = np.array([ACTIVE_LIB] * n_actives, dtype=object)
    titles_fill = np.array([None] * n_actives, dtype=object)
    smiles_fill = np.array([None] * n_actives, dtype=object)
    size_fill = np.zeros(n_actives, dtype=np.int64)

    x2 = np.concatenate([x, x_fill])
    y2 = np.concatenate([y, y_fill])
    labels2 = np.concatenate([labels, labels_fill])
    titles2 = np.concatenate([titles, titles_fill])
    smiles2 = np.concatenate([smiles, smiles_fill])
    cluster_size2 = np.concatenate([cluster_size, size_fill])
    return x2, y2, labels2, titles2, smiles2, cluster_size2

def hexbin_coverage(selected_idx, hb_payload, use_kept=True):
    """
    Fraction of unique hexbins hit by selected points.
    - Numerator: count of unique hex IDs among selected points (optionally restricted to kept)
    - Denominator: total number of hexbins considered (kept or all)
    """
    point_hex_id = np.asarray(hb_payload["point_hex_id"], dtype=int)
    keep_hex = np.asarray(
        hb_payload.get("keep_hex", np.ones_like(hb_payload["hex_mass"], dtype=bool)),
        dtype=bool
    )

    if selected_idx.size == 0:
        return 0.0

    hit_hex = np.unique(point_hex_id[selected_idx])

    if use_kept:
        denom = int(keep_hex.sum())
        num = int(np.sum(keep_hex[hit_hex]))  # count only hits that are in kept
    else:
        denom = int(keep_hex.size)
        num = int(hit_hex.size)

    return (num / denom) if denom > 0 else np.nan


def murcko_count(smiles_list):
    if not RDKit_OK:
        return np.nan
    scaff = set()
    for smi in smiles_list:
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if sc:
            scaff.add(sc)
    return int(len(scaff))


def morgan_bitvects(smiles_list, radius=2, nbits=2048):
    fps = []
    for smi in smiles_list:
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits))
    return fps


def isim_from_bitvects(bitvects, index="JT"):
    """
    Best effort:
    - If iSIM works: compute iSIM on unpacked 0/1 array.
    - Else: fall back to mean pairwise Tanimoto.
    """
    if len(bitvects) < 2:
        return np.nan

    arr = np.asarray([list(map(int, fp.ToBitString())) for fp in bitvects], dtype=np.uint8)
    return float(calculate_isim(arr, n_ary=index))


def mass_coverage_topk_decoys(selected_idx, cluster_size, labels, decoy_lib, hb_payload=None, restrict_to_kept=True, K=None):
    """
    Normalized mass coverage vs the BEST possible mass achievable with K decoys.
      numerator   = sum(cluster_size[selected_idx])
      denominator = sum of top-K cluster_size among eligible decoys

    If hb_payload is provided and restrict_to_kept=True, "eligible" means decoys whose points
    fall in kept hexbins (keep_hex[point_hex_id]).

    Set K to len(selected_idx) (default) or to your target_decoys.
    """
    import numpy as np

    cs = np.asarray(cluster_size, dtype=np.float64)
    labels = np.asarray(labels)

    selected_idx = np.asarray(selected_idx, dtype=int)
    if K is None:
        K = int(selected_idx.size)
    K = int(K)

    if K <= 0:
        return np.nan

    # --- build eligible decoy set ---
    decoy_mask = (labels == decoy_lib)

    if hb_payload is not None and restrict_to_kept:
        point_hex_id = np.asarray(hb_payload["point_hex_id"], dtype=int)
        keep_hex = np.asarray(hb_payload.get("keep_hex", None), dtype=bool)
        if keep_hex is not None:
            kept_points = keep_hex[point_hex_id]
            decoy_mask = decoy_mask & kept_points

    eligible = np.where(decoy_mask)[0]
    if eligible.size == 0:
        return np.nan

    K_eff = min(K, int(eligible.size))

    # --- denominator: sum of top-K_eff masses among eligible decoys ---
    eligible_mass = cs[eligible]
    if K_eff == eligible_mass.size:
        denom = float(eligible_mass.sum())
    else:
        # stable + fast: top-K via partition
        denom = float(np.partition(eligible_mass, -K_eff)[-K_eff:].sum())

    # --- numerator: mass of selected decoys (optionally clipped to eligible set) ---
    # If you want to be strict, only count selected that are eligible:
    # selected_idx = selected_idx[np.isin(selected_idx, eligible)]
    num = float(cs[selected_idx].sum()) if selected_idx.size else 0.0

    print("denominator in mass (top-K eligible decoys): ", denom)
    print("numerator in mass (selected decoys): ", num)

    return num / denom if denom > 0 else np.nan
def main():
    parser = argparse.ArgumentParser(description="Hexbin cap study for benchmarking approaches")
    parser.add_argument("--approach", type=int, choices=[2, 3], default=2)
    parser.add_argument("--npz-path", default=None,
                        help="Optional explicit embedding NPZ. Defaults from approach.")
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED,
                        help="Global random seed used by this cap-study run")
    args = parser.parse_args()
    rng_seed = int(args.rng_seed)
    set_global_seed(rng_seed)

    npz_path = args.npz_path
    if npz_path is None:
        npz_path = NPZ_PATH_APPROACH_2 if args.approach == 2 else NPZ_PATH_APPROACH_3

    # -----------------------------
    # Load data
    # -----------------------------
    data = np.load(npz_path, allow_pickle=True)
    emb = data["embedding"]
    x = emb[:, 0].astype(float)
    y = emb[:, 1].astype(float)
    labels = np.asarray(data["labels"])
    titles = np.asarray(data["titles"], dtype=object)
    smiles = np.asarray(data["smiles"], dtype=object)
    cluster_size = np.asarray(data["cluster_sizes"], dtype=np.int64)
    approach_meta = {}
    if "approach_meta" in data and len(data["approach_meta"]) > 0:
        meta0 = data["approach_meta"][0]
        if isinstance(meta0, dict):
            approach_meta = meta0

    n_act = int(np.sum(labels == ACTIVE_LIB))
    n_dec = int(np.sum(labels == DECOY_LIB))

    if n_act == 0:
        n_act_ref = load_active_count_from_pkl()
        max_feasible_act = n_dec // max(1, int(RATIO_K))
        n_act_eff = min(n_act_ref, max_feasible_act)

        if n_act_eff < n_act_ref:
            print(
                f"[WARN] Capping active count for ratio sizing from {n_act_ref} to {n_act_eff} "
                f"(decoys available={n_dec}, ratio_k={RATIO_K})."
            )
        else:
            print(f"[INFO] No embedded actives found. Using {n_act_ref} actives from PKL for ratio sizing.")

        x, y, labels, titles, smiles, cluster_size = inject_synthetic_actives_for_ratio(
            x, y, labels, titles, smiles, cluster_size, n_act_eff
        )
        n_act = int(np.sum(labels == ACTIVE_LIB))
        n_dec = int(np.sum(labels == DECOY_LIB))

    print(f"Approach={args.approach} | NPZ={npz_path}")
    print(f"Actives(for ratio sizing)={n_act}, Decoys={n_dec}")
    mass_mode = approach_meta.get("mass_mode", None)
    selection_density_mode_meta = approach_meta.get("selection_density_mode", None)
    if selection_density_mode_meta is not None:
        density_mode = selection_density_mode_meta
    else:
        density_mode = "weighted_vs_mass" if args.approach == 3 else "count"

    density_label = "VS weighted mass density" if density_mode == "weighted_vs_mass" else "VS medoid count density"
    print(f"Approach mass mode metadata: {mass_mode}")
    print(f"Selection density mode: {density_mode}")

    if not RDKit_OK:
        print("[WARN] RDKit not importable in this environment -> Murcko will be NaN.")

    # ============================
    # Outer sweep over cap_max
    # ============================
    for cap_max in CAP_MAX_GRID:
        OUTPUT_DIR = f"Approach_{args.approach}_Cap_max_{cap_max}_ratio_{RATIO_K}"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        MID_CAP_GRID = list(range(1, cap_max + 1))

        rows = []

        for mid_cap in MID_CAP_GRID:
            tag = f"midcap_{mid_cap}_capmax_{cap_max}_alpha_{ALPHA}"

            selected_decoys, chosen_nbins, _, hb_payload = hexbin_density_autotune_nbins(
                x, y,
                labels, titles, smiles,
                cluster_size,
                out_png=os.path.join(OUTPUT_DIR, f"{tag}_hexbin.png"),
                decoy_lib=DECOY_LIB,
                active_lib=ACTIVE_LIB,
                decoy_csv_path=os.path.join(OUTPUT_DIR, f"{tag}_decoys.csv"),
                nbins_min=NBINS_MIN,
                nbins_max=NBINS_MAX,
                nbins_step=NBINS_STEP,
                ratio_k=RATIO_K,
                per_hex_cap=mid_cap,
                per_hex_cap_max=cap_max,      # <-- swept cap_max here
                cap_alpha=ALPHA,
                cap_ref_quantile=CAP_REF_Q,
                min_log_mass=MIN_LOG_MASS,
                rng_seed=rng_seed,
                density_mode=density_mode,
            )

            selected_decoys = np.asarray(selected_decoys, dtype=int)

            # (A) hexbin coverage
            hexbin_cov_kept = hexbin_coverage(selected_decoys, hb_payload, use_kept=True)

            # (C) Spearman density adherence
            M_selection = np.asarray(hb_payload["hex_mass"], float)
            S = np.asarray(hb_payload["selected_counts_per_hex"], float)
            keep = np.asarray(hb_payload["keep_hex"], bool)
            point_hex_id = np.asarray(hb_payload["point_hex_id"], dtype=int)

            vs_mask = (labels != DECOY_LIB) & (labels != ACTIVE_LIB)
            M_mass = np.bincount(
                point_hex_id[vs_mask],
                weights=cluster_size[vs_mask],
                minlength=M_selection.size
            ).astype(float)

            mask = keep
            spearman_selection_rho = spearmanr(M_selection[mask], S[mask]).statistic if np.sum(mask) >= 2 else np.nan
            spearman_mass_rho = spearmanr(M_mass[mask], S[mask]).statistic if np.sum(mask) >= 2 else np.nan

            # (B) internal decoy mass coverage (cluster-size based, decoys only)
            mass_cov_kept = mass_coverage_topk_decoys(
                selected_decoys,
                cluster_size,
                labels,
                DECOY_LIB,
                hb_payload=hb_payload,
                restrict_to_kept=True,
                K=selected_decoys.size
            )

            # iSIM + Murcko
            isim_val = np.nan
            murcko_val = np.nan
            if RDKit_OK:
                fps = morgan_bitvects(smiles[selected_decoys].tolist())
                isim_val = isim_from_bitvects(fps, index="JT")
                murcko_val = murcko_count(smiles[selected_decoys].tolist())

            rows.append({
                "Cap_max": cap_max,
                "Mid_cap": mid_cap,
                "chosen_nbins": int(chosen_nbins),
                "n_selected_decoys": int(selected_decoys.size),
                "hexbin_cov_kept": hexbin_cov_kept,
                "mass_cov_kept": mass_cov_kept,
                "density_mode": density_mode,
                "mass_mode": mass_mode,
                "rng_seed": rng_seed,
                "isim_decoys": float(isim_val) if np.isfinite(isim_val) else np.nan,
                "murcko_scaffolds_decoys": murcko_val,
                "spearman_count_density": float(spearman_selection_rho),  # kept for backwards compatibility
                "spearman_selection_density": float(spearman_selection_rho),
                "spearman_mass_density": float(spearman_mass_rho),
            })

            print(
                f"[cap_max={cap_max} mid_cap={mid_cap}] nbins={chosen_nbins} "
                f"n_decoys={selected_decoys.size} "
                f"hexbin_cov={hexbin_cov_kept:.4f} "
                f"mass_cov_internal={mass_cov_kept:.4e} "
                f"spearman_selection={spearman_selection_rho:.4f} "
                f"spearman_mass={spearman_mass_rho:.4f} "
                f"isim={isim_val:.6f} murcko={murcko_val}"
            )

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUTPUT_DIR, f"cap_metrics_capmax_{cap_max}.csv"), index=False)

        # ---- plots per cap_max ----
        plt.figure(); plt.plot(df["Mid_cap"], df["chosen_nbins"], marker="o")
        plt.xlabel("mid_cap"); plt.ylabel("minimal nbins to reach target decoys")
        plt.title(f"mid_cap vs nbins (cap_max = {cap_max})"); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "midcap_vs_nbins.png"), dpi=600); plt.close()

        plt.figure(); plt.plot(df["Mid_cap"], df["isim_decoys"], marker="o")
        plt.axhline(ISIM_GOAL, linestyle="--")
        plt.xlabel("mid_cap"); plt.ylabel("iSIM (decoys-only)")
        plt.title(f"Similarity using iSIM (cap_max = {cap_max})"); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "midcap_vs_isim.png"), dpi=600); plt.close()

        plt.figure(); plt.plot(df["Mid_cap"], df["murcko_scaffolds_decoys"], marker="o")
        plt.xlabel("mid_cap"); plt.ylabel("Unique Murcko scaffolds (selected decoys)")
        plt.title(f"Diversity (cap_max = {cap_max})"); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "midcap_vs_murcko.png"), dpi=600); plt.close()

        plt.figure(); plt.plot(df["Mid_cap"], df["hexbin_cov_kept"], marker="o")
        plt.xlabel("mid_cap"); plt.ylabel("Hexbin area coverage")
        plt.title(f"Space Coverage (cap_max = {cap_max})"); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "midcap_vs_hex_area.png"), dpi=600); plt.close()

        plt.figure(); plt.plot(df["Mid_cap"], df["mass_cov_kept"], marker="o")
        plt.xlabel("mid_cap"); plt.ylabel("Internal decoy mass coverage")
        plt.title(f"Internal Decoy Mass Coverage (cap_max = {cap_max})"); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "midcap_vs_internal_mass_cov.png"), dpi=600); plt.close()

        plt.figure(); plt.plot(df["Mid_cap"], df["spearman_count_density"], marker="o")
        plt.xlabel("mid_cap"); plt.ylabel(f"Spearman rho ({density_label} vs selected decoys per hexbin)")
        plt.title(f"Correlation with Selection Density (cap_max = {cap_max})"); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "midcap_vs_spearman_count.png"), dpi=600); plt.close()

        plt.figure(); plt.plot(df["Mid_cap"], df["spearman_mass_density"], marker="o")
        plt.xlabel("mid_cap"); plt.ylabel("Spearman ρ (hexbin VS mass vs selected decoys per hexbin)")
        plt.title(f"Correlation with Mass Density (cap_max = {cap_max})"); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "midcap_vs_spearman_mass.png"), dpi=600); plt.close()





if __name__ == "__main__":
    main()
