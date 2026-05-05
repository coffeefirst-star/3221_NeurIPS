# Instruction to create datasets with the three approaches

# All of these methods require the initial Libraries clustering to produce the .pkl files containing the medoids.
# Approach 1: re-cluster simple
1. python re-clustering.py --approach 1 --rng-seed 42 --recluster-shuffle 1


# Approach 2: UMAP simple / unweighted
1. python umap_libs.py --approach 2 --output-prefix umap_approach_2 --rng-seed 42
2. python ratio_1_cap_study.py --approach 2 --npz-path umap_approach_2_embedding.npz --rng-seed 42
# once datasets are generated with cap combinations
3. python score.py
# select the best combination of caps
# asess with Mean NN and physicochemical distributions comparison to VS
4. python mean_nn.py
5. python distros_phys.py
6. python plot_cached_descriptor_distributions.py

# Approach 3: UMAP weighted
1. python umap_libs.py --approach 3 --output-prefix umap_approach_3 --rng-seed 52
2. python .ratio_1_cap_study.py --approach 3 --npz-path umap_approach_3_embedding.npz --rng-seed 52
# once datasets are generated with cap combinations
3. python score.py

# Approach 4:
1. python 4-KMeans.py