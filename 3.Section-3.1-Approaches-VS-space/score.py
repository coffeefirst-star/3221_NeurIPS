import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_minmax(s):
    s = pd.to_numeric(s, errors="coerce")
    lo = s.min()
    hi = s.max()
    if pd.isna(lo) or pd.isna(hi):
        return pd.Series(np.nan, index=s.index)
    if hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


glued_data = pd.DataFrame()

if os.path.exists("all_ratio_1.csv"):
    print("all_ratio_1.csv already exists - skipping computation.")
else:
    for file_name in glob.glob("Approach_*_Cap_max_*_ratio_1/cap_metrics_capmax_*.csv"):
        print(file_name)
        x = pd.read_csv(file_name, low_memory=False)
        glued_data = pd.concat([glued_data, x], axis=0)

    glued_data.to_csv("all_ratio_1.csv", index=False)

glued_data = pd.read_csv("all_ratio_1.csv")
glued_data.rename(columns={"Cap_max": "Max_cap"}, inplace=True)

col1 = "hexbin_cov_kept"
col2 = "spearman_count_density"
col3 = "murcko_scaffolds_decoys"

glued_data[f"{col1}_norm"] = safe_minmax(glued_data[col1])
glued_data[f"{col2}_norm"] = safe_minmax(glued_data[col2])
glued_data[f"{col3}_norm"] = safe_minmax(glued_data[col3])

f1 = "hexbin_cov_kept_norm"
f2 = "spearman_count_density_norm"
f3 = "murcko_scaffolds_decoys_norm"

w1, w2, w3 = 0.4, 0.5, 0.10

glued_data["global_score"] = (
    w1 * glued_data[f1]
    + w2 * glued_data[f2]
    + w3 * glued_data[f3]
)

heatmap_df = glued_data.pivot_table(
    index="Max_cap",
    columns="Mid_cap",
    values="global_score",
    aggfunc="max"
)

data = heatmap_df.values.astype(float)
masked = np.ma.masked_invalid(data)

fig, ax = plt.subplots(figsize=(8, 6))
fig.tight_layout(pad=2.2)

im = ax.imshow(masked, origin="lower", aspect="auto", cmap="YlGnBu")
cbar = plt.colorbar(im, ax=ax, label="Global score")
cbar.ax.tick_params(labelsize=12)
cbar.set_label("Global score", fontsize=16)

ax.set_xlabel("Mid_cap", fontsize=16)
ax.set_ylabel("Max_cap", fontsize=16)
ax.set_title(
    r"Global score = $0.4\,HC + 0.5\,SD + 0.10\,MS$",
    fontsize=16
)

ax.set_xticks(np.arange(len(heatmap_df.columns)))
ax.set_xticklabels(heatmap_df.columns)
ax.set_yticks(np.arange(len(heatmap_df.index)))
ax.set_yticklabels(heatmap_df.index)
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=12)

N = 5
valid_coords = np.argwhere(~masked.mask)
flat_vals = masked.compressed()
top_flat_idx = np.argsort(flat_vals)[::-1][:N]
top_coords = valid_coords[top_flat_idx]

for i, j in top_coords:
    val = f"{data[i, j]:.2f}"
    if val.startswith("0."):
        val = val[1:]
    ax.text(j, i, val, ha="center", va="center", fontsize=11, fontweight="medium", color="white")

plt.savefig("global_score_heatmap.png", dpi=800, bbox_inches="tight", pad_inches=0.02)

