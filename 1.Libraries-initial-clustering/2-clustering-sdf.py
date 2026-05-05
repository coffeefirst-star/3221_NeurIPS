
import argparse
import pickle
import numpy as np
import bblean
import bblean.similarity as iSIM
import os 
import time
# ==== CONFIG ====
FP_DIR = "./"    # folder with your .npy fingerprint chunks
N_FEATURES = 2048             # fingerprint length
OUTPUT_NAME = "npy_medoids.pkl"
# =================

# ---- 1. Load all .npy fingerprint chunks ----
print(f"Loading .npy fingerprint chunks from {FP_DIR}...")
fps_list = []
n_mols = 0
for fname in sorted(os.listdir(FP_DIR)):
    if not fname.endswith(".npy"):
        continue
    path = os.path.join(FP_DIR, fname)
    print(f"  ↳ loading {fname}")
    arr = np.load(path, mmap_mode="r")  # efficient memory mapping
    fps_list.append(arr)
    n_mols += arr.shape[0]

print(f"Loaded {len(fps_list)} chunks, total {n_mols:,} molecules.")


# Concatenate (keep mmap if memory allows)
fps = np.concatenate(fps_list, axis=0)
del fps_list

# ---- 2. Select optimal threshold ----
if len(fps) > 10_000_000:
    random_sample = np.random.choice(len(fps), size=1_000_000, replace=False)
    fps_sample = fps[random_sample]
    representative_samples = iSIM.jt_stratified_sampling(fps_sample, n_samples=50)
    representative_samples = random_sample[representative_samples]
    del fps_sample
else:
    representative_samples = iSIM.jt_stratified_sampling(fps, n_samples=50)

sim_matrix = iSIM.jt_sim_matrix_packed(fps[representative_samples])
sim_matrix = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]
average_sim = np.mean(sim_matrix)
std = np.std(sim_matrix)
del sim_matrix

optimal_threshold = average_sim + 3.5 * std
print(f"Optimal threshold = {optimal_threshold:.4f}")

# ---- 3. BitBirch clustering ----
start = time.time()
bb_tree = bblean.BitBirch(
    branching_factor=50,
    threshold=optimal_threshold,
    merge_criterion="diameter"
)
bb_tree.fit(fps)


print(f"initial bitbirch took {time.time() - start:.2f} s")

# ---- 4. Refinement ----
start = time.time()
bb_tree.recluster_inplace(
    iterations=5,
    extra_threshold=std,
    shuffle=False,
    verbose=True
)
print(f"refinement took {time.time() - start:.2f} s")

# ---- 5. Extract clusters and medoids ----
import pickle as pkl
clusters = bb_tree.get_cluster_mol_ids()

with open('clustered_ids_parallel.pkl', 'wb') as f:
    pkl.dump(clusters, f)
    
cluster_size = []
fingerprints_medoids = []
medoid_indices = []
medoid_fps_packed = []  # NEW

for cluster in clusters:
    cluster_size.append(len(cluster))
    fps_cluster = fps[cluster]
    medoid_id, medoid_fp = iSIM.jt_isim_medoid(
        fps_cluster,
        input_is_packed=True,
        n_features=N_FEATURES,
        pack=True
    )
    fingerprints_medoids.append(bblean.unpack_fingerprints(medoid_fp))
    medoid_indices.append(cluster[medoid_id])
    medoid_fps_packed.append(medoid_fp)  # NEW


# ---- 6. Save results ----
with open(OUTPUT_NAME, "wb") as f:
    pickle.dump(
        {
            "fingerprints": fingerprints_medoids,
            "medoid_indices": medoid_indices,
            "cluster_size": cluster_size,
        },
        f,
    )

print(f"✅ Saved {len(fingerprints_medoids):,} medoids to {OUTPUT_NAME}")


