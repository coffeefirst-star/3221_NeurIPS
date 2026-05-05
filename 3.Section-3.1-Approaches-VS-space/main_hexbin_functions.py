#!/usr/bin/env python3
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import csv
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import pandas as pd
import plotly.express as px
import plotly.colors as pc
import matplotlib.patheffects as pe

DECOY_LIB = "UF-Scripps-Decoys"
ACTIVE_LIB = "UF-Scripps-Actives"
NPZ_PATH = "umap_medoids_all_libraries_embedding.npz"
SCATTER_OUT = "umap_medoids_with_marginals.png"
DENSITY_OUT = "umap_medoids_hexbin_density.png"
SCATTER3D_OUT = "umap_medoids_3d.png"

MAX_PER_LIB = 500000000
RNG_SEED = 42

def set_global_seed(seed):
    """
    Set deterministic RNG state for modules used in this script.
    """
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)


def stratified_subsample(embedding, labels, max_per_lib, seed):
    """
    Works for 2D or 3D embeddings.
    For 3D we will use x = comp1, y = comp2, z = comp3.
    """
    x = embedding[:, 0]
    y = embedding[:, 1]
    z = embedding[:, 2] if embedding.shape[1] > 2 else None

    unique_libs, counts = np.unique(labels, return_counts=True)
    rng = np.random.default_rng(seed)
    idx_list = []

    for lib, count in zip(unique_libs, counts):
        mask = labels == lib
        idx_lib = np.where(mask)[0]
        if count > max_per_lib:
            idx_sel = rng.choice(idx_lib, size=max_per_lib, replace=False)
        else:
            idx_sel = idx_lib
        idx_list.append(idx_sel)
    idx = np.concatenate(idx_list)

    idx = np.concatenate(idx_list)
    x_sub = x[idx]
    y_sub = y[idx]
    z_sub = z[idx] if z is not None else None
    labels_sub = labels[idx]
    return x_sub, y_sub, z_sub, labels_sub, unique_libs, idx

def stacked_marginal_histograms(ax_histx, ax_histy, x, y, labels, unique_libs, lib_to_color, nbins=120):
    """
    Build stacked histograms in X and Y so each bin shows composition by library.
    """
    x_min, x_max = x.min(), x.max()
    bins_x = np.linspace(x_min, x_max, nbins + 1)
    bin_centers_x = 0.5 * (bins_x[:-1] + bins_x[1:])
    bin_width_x = bins_x[1] - bins_x[0]
    counts_x = []
    for lib in unique_libs:
        mask = labels == lib
        hist, _ = np.histogram(x[mask], bins=bins_x)
        counts_x.append(hist)
    counts_x = np.asarray(counts_x) 
    bottom = np.zeros_like(bin_centers_x, dtype=float)
    for lib, hist in zip(unique_libs, counts_x):
        ax_histx.bar(
            bin_centers_x,
            hist,
            width=bin_width_x,
            bottom=bottom,
            color=lib_to_color[lib],
            alpha=0.85,
            linewidth=0,
        )
        bottom += hist
    ax_histx.set_ylabel("Count")
    ax_histx.tick_params(axis="x", labelbottom=False)
    y_min, y_max = y.min(), y.max()
    bins_y = np.linspace(y_min, y_max, nbins + 1)
    bin_centers_y = 0.5 * (bins_y[:-1] + bins_y[1:])
    bin_height_y = bins_y[1] - bins_y[0]
    counts_y = []
    for lib in unique_libs:
        mask = labels == lib
        hist, _ = np.histogram(y[mask], bins=bins_y)
        counts_y.append(hist)
    counts_y = np.asarray(counts_y)
    left = np.zeros_like(bin_centers_y, dtype=float)
    for lib, hist in zip(unique_libs, counts_y):
        ax_histy.barh(
            bin_centers_y,
            hist,
            height=bin_height_y,
            left=left,
            color=lib_to_color[lib],
            alpha=0.85,
            linewidth=0,
        )
        left += hist
    ax_histy.set_xlabel("Count")
    ax_histy.tick_params(axis="y", labelleft=False)

def scatter_with_marginals(x, y, labels, unique_libs, out_png):
    cmap = plt.get_cmap("tab20")
    lib_to_color = {lib: cmap(i % cmap.N) for i, lib in enumerate(unique_libs)}
    fig = plt.figure(figsize=(10, 9))
    gs = gridspec.GridSpec(4, 4, wspace=0.05, hspace=0.05)
    ax_scatter = fig.add_subplot(gs[1:, :3])
    ax_histx = fig.add_subplot(gs[0, :3], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[1:, 3], sharey=ax_scatter)
    _, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts, kind="mergesort")[::-1]
    for k in order:
        lib = unique_libs[k]
        mask = labels == lib
        if not np.any(mask):
            continue
        ax_scatter.scatter(
            x[mask],
            y[mask],
            s=6.0,
            c=[lib_to_color[lib]],
            alpha=0.6,
            edgecolors="none",
            label=lib,
        )
    ax_scatter.set_xlabel("UMAP 1")
    ax_scatter.set_ylabel("UMAP 2")
    ax_scatter.set_title("UMAP of BitBirch medoids across libraries")
    stacked_marginal_histograms(ax_histx, ax_histy, x, y, labels, unique_libs, lib_to_color, nbins=120)
    ax_scatter.legend(
        markerscale=1.5, fontsize=7, frameon=True, loc="upper right", ncol=1
    )

    plt.savefig(out_png, dpi=600)
    plt.close()
    print(f"Saved {out_png}")


def select_nearest_active_decoys(x, y, labels, allowed_mask, decoy_lib, active_lib, analog_target):
    if analog_target <= 0:
        return np.array([], dtype=int)

    labels = np.asarray(labels)
    decoy_idx = np.where((labels == decoy_lib) & allowed_mask)[0]
    active_idx = np.where(labels == active_lib)[0]
    if decoy_idx.size == 0 or active_idx.size == 0:
        return np.array([], dtype=int)

    tree = cKDTree(np.column_stack([x[active_idx], y[active_idx]]))
    distances, _ = tree.query(np.column_stack([x[decoy_idx], y[decoy_idx]]), k=1)
    order = np.argsort(distances, kind="mergesort")
    return decoy_idx[order[: min(int(analog_target), decoy_idx.size)]].astype(int)

def hexbin_density_autotune_nbins(
    x, y,
    labels, titles, smiles,
    cluster_size,
    allowed_mask=None,
    out_png=None,
    decoy_lib=None,
    active_lib=None,
    decoy_csv_path=None,
    nbins_min=50,
    nbins_max=200,
    nbins_step=5,
    ratio_k=5,
    per_hex_cap=1,
    per_hex_cap_max=None,
    cap_alpha=0.5,
    cap_ref_quantile=0.5,
    rng_seed=42,
    min_log_mass=1.0,
    density_mode="count",
    analog_fraction=0.25,
    analog_mode="nearest_active_umap",
    dpi=600,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    import csv
    if allowed_mask is None:
        allowed_mask = np.ones(len(labels), dtype=bool)
    if density_mode not in {"count", "weighted_vs_mass"}:
        raise ValueError("density_mode must be one of {'count', 'weighted_vs_mass'}")

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    labels = np.asarray(labels)
    cluster_size = np.asarray(cluster_size, np.float64)

    decoy_mask = labels == decoy_lib
    active_mask = labels == active_lib
    vs_mask = ~(decoy_mask | active_mask)  # <-- VS-only definition (permanent)

    n_act = int(active_mask.sum())
    if n_act == 0:
        raise ValueError("No actives found.")
    target_decoys = int(ratio_k * n_act)
    analog_target = int(round(target_decoys * float(np.clip(analog_fraction, 0.0, 1.0))))

    rng = np.random.default_rng(rng_seed)

    def assign_hex_id(centers, pts):
        tree = cKDTree(centers)
        return tree.query(pts, k=1)[1].astype(int)

    def try_nbins(nbins):
        # Build hexbin on VS points; values are either count-per-hex or weighted VS mass.
        fig, ax = plt.subplots()
        if density_mode == "weighted_vs_mass":
            hb = ax.hexbin(
                x[vs_mask], y[vs_mask],
                C=cluster_size[vs_mask],
                reduce_C_function=np.sum,
                gridsize=int(nbins),
                mincnt=1,
                norm=LogNorm(),
            )
        else:
            hb = ax.hexbin(
                x[vs_mask], y[vs_mask],
                gridsize=int(nbins),
                mincnt=1,
                norm=LogNorm(),
            )
        centers = hb.get_offsets()
        hex_mass = np.asarray(hb.get_array(), dtype=float)
        plt.close(fig)

        if hex_mass.size == 0:
            return False, None

        log_mass = np.log10(hex_mass)
        keep_hex = log_mass >= float(min_log_mass)
        
        if not np.any(keep_hex):
            return False, None

        # Assign ALL points (including decoys) to the VS-defined hex centers
        point_hex_id = assign_hex_id(centers, np.column_stack([x, y]))

        # Eligible decoys must fall into kept hexbins
        decoy_idx = np.where((labels == decoy_lib) & allowed_mask)[0]
        if decoy_idx.size == 0:
            return False, None

        analog_selected = np.array([], dtype=int)
        if analog_mode == "nearest_active_umap" and analog_target > 0:
            analog_selected = select_nearest_active_decoys(
                x, y, labels, allowed_mask, decoy_lib, active_lib, analog_target
            )

        # Cap based on the selected density definition.
        usable_mass = hex_mass[keep_hex]
        ref_mass = max(np.quantile(usable_mass, float(cap_ref_quantile)), 1.0)

        cap = float(per_hex_cap) * (hex_mass / ref_mass) ** float(cap_alpha)
        cap = np.maximum(1, np.rint(cap))
        if per_hex_cap_max is not None:
            cap = np.clip(cap, 1, int(per_hex_cap_max))
        cap[~keep_hex] = 0
        cap = cap.astype(int)

        hx = point_hex_id[decoy_idx]
        decoy_counts_per_hex = np.bincount(hx, minlength=hex_mass.size)

        # Greedy ranking: prioritize decoys in high-density bins.
        order = np.argsort(-hex_mass[hx], kind="mergesort")
        ranked = decoy_idx[order]
        if analog_selected.size:
            ranked = ranked[~np.isin(ranked, analog_selected)]

        counts = np.zeros_like(hex_mass, dtype=int)
        selected = []
        remaining_target = target_decoys - int(analog_selected.size)
        for idx in ranked:
            h = point_hex_id[idx]
            if counts[h] < cap[h]:
                selected.append(int(idx))
                counts[h] += 1
                if len(selected) >= remaining_target:
                    break

        selected = np.asarray(selected, dtype=int)
        if analog_selected.size:
            selected = np.concatenate([analog_selected, selected]).astype(int)
        if selected.size < target_decoys:
            return False, None

        payload = dict(
            centers=centers,
            hex_mass=hex_mass,                # VS-only
            keep_hex=keep_hex,
            point_hex_id=point_hex_id,
            cap_per_hex=cap,
            decoy_counts_per_hex=decoy_counts_per_hex,
            selected_counts_per_hex=counts.copy(),
            vs_mask=vs_mask,                  # handy for downstream plotting
            density_mode=density_mode,
            analog_selected=analog_selected.astype(int),
        )
        return True, (selected, payload)

    # --- sweep nbins ---
    chosen_nbins = None
    chosen_selected = None
    chosen_payload = None

    for nbins in range(nbins_min, nbins_max + 1, nbins_step):
        ok, result = try_nbins(nbins)
        if ok:
            chosen_selected, chosen_payload = result
            chosen_nbins = int(nbins)
            break

    if chosen_nbins is None:
        raise RuntimeError(
            "No nbins satisfied target_decoys — try increasing per_hex_cap / CAP_MAX, "
            "lowering ratio_k, or expanding nbins_max."
        )

    # --- correlation (ONCE, kept only) ---
    keep_hex = chosen_payload["keep_hex"]
    point_hex_id = chosen_payload["point_hex_id"]
    decoy_counts_per_hex = chosen_payload["decoy_counts_per_hex"]

    selection_density = np.asarray(chosen_payload["hex_mass"], dtype=float)
    rho, pval = spearmanr(decoy_counts_per_hex[keep_hex], selection_density[keep_hex])
    print(f"[nbins={chosen_nbins}] Spearman ({density_mode}, kept only): rho={rho:.3f}, p={pval:.2e}")

    # --- final plot (VS-only density heatmap) ---
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    hex_mass = chosen_payload["hex_mass"]
    x_all = x[vs_mask]
    y_all = y[vs_mask]

    xlo, xhi = np.percentile(x_all, [0.05, 99.95])
    ylo, yhi = np.percentile(y_all, [0.15, 99.85])

    # small padding so points don't touch the frame
    px = 0.2 * (xhi - xlo)
    py = 0.2 * (yhi - ylo)
    if density_mode == "weighted_vs_mass":
        hb = ax.hexbin(
            x_all, y_all,
            C=cluster_size[vs_mask],
            reduce_C_function=np.sum,
            gridsize=chosen_nbins,
            mincnt=1,
            norm=LogNorm(vmin=1, vmax=np.percentile(hex_mass, 100)),
            cmap="YlGnBu",
            linewidths=0.0,
        )
    else:
        hb = ax.hexbin(
            x_all, y_all,
            gridsize=chosen_nbins,
            mincnt=1,
            norm=LogNorm(vmin=1, vmax=np.percentile(hex_mass, 100)),
            cmap="YlGnBu",
            linewidths=0.0,
        )

    # overlays
    sel_counts   = np.asarray(chosen_payload["selected_counts_per_hex"], dtype=int)
    decoy_counts = np.asarray(chosen_payload["decoy_counts_per_hex"], dtype=int)
    centers      = np.asarray(chosen_payload["centers"])
    cap_per_hex  = np.asarray(chosen_payload["cap_per_hex"], dtype=int)

    # -----------------------------
    # Hybrid labeling configuration
    # -----------------------------
    # -----------------------------
    # Hybrid labeling configuration
    # -----------------------------
    K_INFO = 15          # informative labels (cap-limited / missed)
    K_REPR = 15          # representative labels (radial stratified)
    SAT_DEC_MIN = 30     # only label saturated bins if decoy_counts >= this

    # -----------------------------
    # 1) Informative bins
    # -----------------------------
    miss_idx = np.where(keep_hex & (sel_counts == 0) & (decoy_counts > 0))[0]
    sat_idx  = np.where(keep_hex & (sel_counts == cap_per_hex) & (decoy_counts >= SAT_DEC_MIN))[0]

    # take top saturated bins by decoy_counts, after including missed bins
    if sat_idx.size:
        sat_idx = sat_idx[np.argsort(decoy_counts[sat_idx], kind="mergesort")[::-1]]

    n_sat_take = max(0, K_INFO - miss_idx.size)
    info_idx = np.unique(np.concatenate([miss_idx, sat_idx[:n_sat_take]]))
    if info_idx.size > K_INFO:
        info_idx = info_idx[:K_INFO]

    # -----------------------------
    # 2) Representative bins (radial stratified)
    # -----------------------------
    cand_idx = np.where(keep_hex & (sel_counts >= 1))[0]
    if info_idx.size and cand_idx.size:
        cand_idx = np.setdiff1d(cand_idx, info_idx, assume_unique=False)

    picked_repr = np.array([], dtype=int)
    if cand_idx.size:
        cand_centers = centers[cand_idx]
        c0 = cand_centers.mean(axis=0)
        r = np.sqrt(((cand_centers - c0) ** 2).sum(axis=1))

        q1, q2 = np.quantile(r, [1/3, 2/3])

        inner = cand_idx[r <= q1]
        mid   = cand_idx[(r > q1) & (r <= q2)]
        outer = cand_idx[r > q2]

        # allocate picks roughly evenly across bands, total = K_REPR
        base = K_REPR // 3
        rem  = K_REPR - 3 * base
        n_inner = base + (1 if rem > 0 else 0)
        n_mid   = base + (1 if rem > 1 else 0)
        n_outer = base

        picked_parts = []
        if inner.size:
            picked_parts.append(rng.choice(inner, size=min(n_inner, inner.size), replace=False))
        if mid.size:
            picked_parts.append(rng.choice(mid, size=min(n_mid, mid.size), replace=False))
        if outer.size:
            picked_parts.append(rng.choice(outer, size=min(n_outer, outer.size), replace=False))

        if picked_parts:
            picked_repr = np.unique(np.concatenate(picked_parts))

    # -----------------------------
    # 3) Final annotation mask + draw
    # -----------------------------
    final_ids = np.unique(np.concatenate([info_idx, picked_repr]))
    mask_annot = np.zeros_like(keep_hex, dtype=bool)
    mask_annot[final_ids] = True

    for (cx, cy), s, d in zip(centers[mask_annot], sel_counts[mask_annot], decoy_counts[mask_annot]):
        if s == 0:
            continue  # skip 0/x labels entirely

        t = ax.text(
            cx, cy, f"{int(s)}/{int(d)}",
            ha="center", va="center",
            fontsize=8,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.025",
                facecolor="white",
                edgecolor="none",
                alpha=0.4
            ),
            zorder=10
        )
        t.set_path_effects([pe.Stroke(linewidth=0.75, foreground="white"), pe.Normal()])

    cbar = fig.colorbar(hb, ax=ax, shrink=0.75, fraction=0.045, pad=0.02, aspect=25)
    if density_mode == "weighted_vs_mass":
        cbar.set_label("VS Represented Mass (log scale)", fontsize=14)
    else:
        cbar.set_label("Number of VS Medoids (log scale)", fontsize=14)
    cbar.ax.tick_params(labelsize=11)
    """
    ax.scatter(
        x[active_mask], y[active_mask],
        marker="x", s=10,
        color="red", linewidths=0.9, alpha=0.6,
        zorder=5,
        label=f"{active_lib}"
    )
"""

    ax.scatter(
        x[chosen_selected], y[chosen_selected],
        marker="o", s=10,
        facecolors="white", edgecolors="black",
        linewidths=0.6, alpha=0.85,
        zorder=6,
        label=f"Selected {decoy_lib}"
    )


    ax.set_xlabel("UMAP 1", fontsize=14)
    ax.set_ylabel("UMAP 2", fontsize=14)
    ax.set_xlim(xlo - px, xhi + px)
    ax.set_ylim(ylo - py, yhi + py)
    ax.set_title(
        f"Hex-UMAP Density-Based Decoy Selection "
        f"(nbins={chosen_nbins}, cap_med={per_hex_cap}, cap_max={per_hex_cap_max})",
        fontsize=14
    )
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    ax.set_aspect("equal", adjustable="datalim")
    plt.savefig(out_png, dpi=dpi)
    plt.close()

    # --- count-based coverage over eligible kept decoys ---
    pid = chosen_payload["point_hex_id"]
    keep_hex = chosen_payload["keep_hex"]
    eligible_decoy_mask = decoy_mask & keep_hex[pid] & allowed_mask
    denom = int(np.sum(eligible_decoy_mask))
    num = int(chosen_selected.size)
    mass_cov = (num / denom) if denom > 0 else np.nan
    print("count_cov:", mass_cov)

    # --- write decoy CSV (fixed) ---
    with open(decoy_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Title", "SMILES"])
        for i in chosen_selected:
            w.writerow([titles[i], smiles[i]])

    return chosen_selected, chosen_nbins, mass_cov, chosen_payload

def scatter3d_plotly(x, y, z, labels, titles, smiles, unique_libs,
                     out_html, out_png=None):
    df = pd.DataFrame({
        "UMAP1": x,
        "UMAP2": y,
        "UMAP3": z,
        "Library": labels,
        "Title": titles,
        "SMILES": smiles,
    })
    base_palette = pc.qualitative.Dark24
    if len(unique_libs) > len(base_palette):
        repeats = (len(unique_libs) // len(base_palette)) + 1
        full_palette = (base_palette * repeats)[:len(unique_libs)]
    else:
        full_palette = base_palette[:len(unique_libs)]
    lib_to_color = {lib: full_palette[i] for i, lib in enumerate(unique_libs)}
    fig = px.scatter_3d(
        df,
        x="UMAP1",
        y="UMAP2",
        z="UMAP3",
        color="Library",
        color_discrete_map=lib_to_color,
        opacity=0.25,
        size_max=5,
        hover_data=["Library", "Title", "SMILES"],
        title="3D UMAP of BitBirch medoids across libraries",
        width=1200,
        height=1100,
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        legend=dict(itemsizing="constant", font=dict(size=14), itemwidth=40)
    )
    fig.write_html(out_html)
    print(f"Saved interactive HTML to {out_html}")
    if out_png is not None:
        fig.write_image(out_png, scale=3)
        print(f"Saved PNG to {out_png}")

def main():
    set_global_seed(RNG_SEED)
    data = np.load(NPZ_PATH, allow_pickle=True)
    embedding = data["embedding"]
    labels = data["labels"]
    titles = data["titles"]
    smiles = data["smiles"]
    cluster_size = data["cluster_sizes"]
    print("\n=== DATA PROVENANCE CHECK ===")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for lib, cnt in zip(unique_labels, counts):
        print(f"{lib:30s} : {cnt:7d}")
    print("\nActive label expected:", ACTIVE_LIB)
    print("Decoy  label expected:", DECOY_LIB)
    print("Detected actives:", np.sum(labels == ACTIVE_LIB))
    print("Detected decoys :", np.sum(labels == DECOY_LIB))
    print("Total points    :", len(labels))
    print("================================\n")
    x_all = embedding[:, 0]
    y_all = embedding[:, 1]
    x_sub, y_sub, z_sub, labels_sub, unique_libs, indices_sub = stratified_subsample(
        embedding, labels, MAX_PER_LIB, RNG_SEED
    )
    scatter_with_marginals(x_sub, y_sub, labels_sub, unique_libs, SCATTER_OUT)
    selected_decoys, chosen_nbins, coverage, hb_payload = hexbin_density_autotune_nbins(
        x_all, y_all,
        labels, titles, smiles,
        cluster_size,
        out_png="umap_medoids_hexbin_autotuned.png",
        decoy_lib=DECOY_LIB,
        active_lib=ACTIVE_LIB,
        decoy_csv_path="dense_decoys_stratified_autotuned.csv",
        nbins_min=30,
        nbins_max=200,
        nbins_step=5,
        ratio_k=5,
        per_hex_cap=1,
        rng_seed=RNG_SEED,
    )
    if z_sub is not None:
        scatter3d_plotly(
            x_sub,
            y_sub,
            z_sub,
            labels_sub,
            titles[indices_sub],
            smiles[indices_sub],
            unique_libs,
            out_html="umap_medoids_3d.html"
        )

if __name__ == "__main__":
    main()
