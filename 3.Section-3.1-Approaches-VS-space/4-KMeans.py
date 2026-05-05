import os
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

BASE_DIR = "../../Libraries"
PKL_NAME = "npy_medoids.pkl"
DECOY_LIB = "UF-Scripps-Decoys"
ACTIVE_LIB = "UF-Scripps-Actives"
K = 1621
RANDOM_STATE = 42
BATCH_SIZE = 4096
N_INIT = 10


def load_all_medoids(base_dir, pkl_name):
    all_fps, all_labels, all_titles, all_smiles = [], [], [], []

    libraries = [
        d for d in sorted(os.listdir(base_dir))
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    for lib in libraries:
        pkl_path = os.path.join(base_dir, lib, pkl_name)
        if not os.path.isfile(pkl_path):
            continue

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        fps = np.asarray(data["fingerprints"], dtype=np.float32)
        titles = np.asarray(data.get("titles_medoids", [None] * len(fps)), dtype=object)
        smiles = np.asarray(data.get("smiles_medoids", [None] * len(fps)), dtype=object)
        labels = np.array([lib] * len(fps), dtype=object)

        all_fps.append(fps)
        all_titles.append(titles)
        all_smiles.append(smiles)
        all_labels.append(labels)

    X = np.concatenate(all_fps, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    titles = np.concatenate(all_titles, axis=0)
    smiles = np.concatenate(all_smiles, axis=0)
    return X, labels, titles, smiles


def squared_l2(a, b):
    diff = a - b
    return np.einsum("ij,ij->i", diff, diff)


X, labels, titles, smiles = load_all_medoids(BASE_DIR, PKL_NAME)

# Keep VS medoids + all decoys, exclude actives
mask = labels != ACTIVE_LIB
X = X[mask]
labels = labels[mask]
titles = titles[mask]
smiles = smiles[mask]

print("Input molecules:", len(X))
print(f"Running MiniBatchKMeans with K={K}, batch_size={BATCH_SIZE}, n_init={N_INIT}")

kmeans = MiniBatchKMeans(
    n_clusters=K,
    random_state=RANDOM_STATE,
    batch_size=BATCH_SIZE,
    n_init=N_INIT,
    reassignment_ratio=0.01,
)
assignments = kmeans.fit_predict(X)

rows = []
cluster_infos = []

for cid in range(K):
    member_idx = np.where(assignments == cid)[0]
    if len(member_idx) == 0:
        continue

    centroid = kmeans.cluster_centers_[cid].reshape(1, -1)
    member_labels = labels[member_idx]
    cluster_decoy_mask = member_labels == DECOY_LIB
    cluster_decoy_idx = member_idx[cluster_decoy_mask]
    vs_mask = member_labels != DECOY_LIB

    primary_decoy_idx = None
    ordered_cluster_decoys = np.array([], dtype=int)
    if cluster_decoy_idx.size > 0:
        cluster_decoy_X = X[cluster_decoy_idx]
        dists = squared_l2(cluster_decoy_X, centroid)
        order = np.argsort(dists, kind="mergesort")
        ordered_cluster_decoys = cluster_decoy_idx[order].astype(int)
        primary_decoy_idx = int(ordered_cluster_decoys[0])

    cluster_n_medoids = int(len(member_idx))
    vs_medoids = int(np.sum(vs_mask))
    decoy_medoids = int(cluster_decoy_idx.size)
    row = {
        "cluster_id": cid,
        "n_members": cluster_n_medoids,
        "cluster_n_medoids": cluster_n_medoids,
        "available_decoys": decoy_medoids,
        "vs_medoids": vs_medoids,
        "decoy_medoids": decoy_medoids,
        "is_decoy_only": bool(np.all(cluster_decoy_mask)),
        "has_decoys": bool(cluster_decoy_idx.size > 0),
        "selected_decoy_idx": primary_decoy_idx,
        "selected_decoy_title": titles[primary_decoy_idx] if primary_decoy_idx is not None else None,
        "selected_decoy_smiles": smiles[primary_decoy_idx] if primary_decoy_idx is not None else None,
    }
    rows.append(row)
    cluster_infos.append({
        "cluster_id": cid,
        "member_idx": member_idx,
        "cluster_n_medoids": cluster_n_medoids,
        "ordered_cluster_decoys": ordered_cluster_decoys,
        "primary_decoy_idx": primary_decoy_idx,
    })

cluster_df = pd.DataFrame(rows).sort_values("cluster_n_medoids", ascending=False).reset_index(drop=True)
cluster_df["rank"] = np.arange(1, len(cluster_df) + 1)
cluster_df = cluster_df[[
    "cluster_id",
    "rank",
    "cluster_n_medoids",
    "available_decoys",
    "vs_medoids",
    "decoy_medoids",
    "is_decoy_only",
    "has_decoys",
    "selected_decoy_idx",
    "selected_decoy_title",
    "selected_decoy_smiles",
]]
cluster_df.to_csv("minibatch_kmeans_1621_cluster_summary.csv", index=False)

# Build final benchmarking set in rounds over clusters sorted from largest to smallest.
# Round 1 takes one decoy from each cluster (if available), round 2 takes a second decoy
# from each cluster, and so on until the quota is filled.
selected_decoy_indices = []
used_decoys = set()
cluster_infos_sorted = sorted(
    cluster_infos,
    key=lambda info: info["cluster_n_medoids"],
    reverse=True,
)

round_idx = 0
while len(selected_decoy_indices) < K:
    added_this_round = False
    for info in cluster_infos_sorted:
        ordered = info["ordered_cluster_decoys"]
        if round_idx >= len(ordered):
            continue
        candidate = int(ordered[round_idx])
        if candidate in used_decoys:
            continue
        selected_decoy_indices.append(candidate)
        used_decoys.add(candidate)
        added_this_round = True
        if len(selected_decoy_indices) >= K:
            break

    if not added_this_round:
        break
    round_idx += 1

if len(selected_decoy_indices) < K:
    raise RuntimeError(
        f"Only {len(selected_decoy_indices)} unique decoys could be selected from clusters with decoys; "
        f"cannot reach quota K={K}."
    )

selected_decoy_indices = np.asarray(selected_decoy_indices[:K], dtype=int)
selected_assignments = assignments[selected_decoy_indices]
selected_cluster_summary = cluster_df.set_index("cluster_id").loc[selected_assignments].reset_index()

ml_ready_df = pd.DataFrame({
    "Molecule Name": titles[selected_decoy_indices],
    "SMILES": smiles[selected_decoy_indices],
    "Label": np.zeros(len(selected_decoy_indices), dtype=int),
    "Source Library": np.full(len(selected_decoy_indices), DECOY_LIB, dtype=object),
    "cluster_id": selected_cluster_summary["cluster_id"].to_numpy(),
    "cluster_n_medoids": selected_cluster_summary["cluster_n_medoids"].to_numpy(),
    "vs_medoids": selected_cluster_summary["vs_medoids"].to_numpy(),
    "available_decoys": selected_cluster_summary["available_decoys"].to_numpy(),
    "decoy_medoids": selected_cluster_summary["decoy_medoids"].to_numpy(),
})
ml_ready_df.to_csv("minibatch_kmeans_1621_ml_ready.csv", index=False)

with open("minibatch_kmeans_1621_medoids.pkl", "wb") as f:
    pickle.dump(
        {
            "selected_indices": selected_decoy_indices,
            "fingerprints": X[selected_decoy_indices],
            "cluster_size": selected_cluster_summary["cluster_n_medoids"].to_numpy(),
            "vs_medoids": selected_cluster_summary["vs_medoids"].to_numpy(),
            "available_decoys": selected_cluster_summary["available_decoys"].to_numpy(),
            "decoy_medoids": selected_cluster_summary["decoy_medoids"].to_numpy(),
            "labels": labels[selected_decoy_indices],
            "titles": titles[selected_decoy_indices],
            "smiles": smiles[selected_decoy_indices],
            "assignments": assignments,
            "selected_assignments": selected_assignments,
            "method": "MiniBatchKMeans",
            "batch_size": BATCH_SIZE,
            "n_init": N_INIT,
            "selection_policy": "round_robin_one_per_cluster_largest_to_smallest",
        },
        f,
    )

print(f"Built {len(selected_decoy_indices)} decoy representatives")
print(f"Clusters with decoys: {int(cluster_df['has_decoys'].sum())}")
print(f"Decoy-only clusters: {int(cluster_df['is_decoy_only'].sum())}")
print(f"Unique selected decoys: {len(np.unique(selected_decoy_indices))}")
print("Saved minibatch_kmeans_1621_cluster_summary.csv, minibatch_kmeans_1621_ml_ready.csv, and minibatch_kmeans_1621_medoids.pkl")
