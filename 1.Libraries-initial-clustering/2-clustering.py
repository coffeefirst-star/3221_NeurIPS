import argparse
import pickle
import numpy as np
import bblean
import bblean.similarity as iSIM
import glob
import time
import sys

log_file = open("clustering.log", "w")
sys.stdout = log_file
sys.stderr = log_file


OUTPUT_NAME = "npy_medoids.pkl"

files = sorted(glob.glob("./*.smi"))
print("Found", len(files), "SMI files")


smiles = bblean.load_smiles(files)
print("Number of SMILES:", len(smiles))

fps, invalid_smiles = bblean.fps_from_smiles(smiles, 
                                             pack=True, 
                                             skip_invalid=True, 
                                             n_features=2048, 
                                             kind="ecfp4")
smiles = np.delete(smiles, invalid_smiles, axis=0)
assert len(smiles) == len(fps), "Number of SMILES and fingerprints do not match!"
print("Number of generated fps:", len(fps))

# Select the optimal threshold
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

# Do the initial clustering
start = time.time()
bb_tree = bblean.BitBirch(branching_factor=50, threshold=optimal_threshold, merge_criterion="diameter")
bb_tree.fit(fps)
print(f"initial bitbirch took {time.time() - start:.2f} s")
# Refine to obtain better clusters
start = time.time()
bb_tree.recluster_inplace(iterations=5, extra_threshold=std, shuffle=False, verbose=True)
print(f"refinement took {time.time() - start:.2f} s")
# Obtain final output
clusters = bb_tree.get_cluster_mol_ids()

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
        n_features=2048,
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


