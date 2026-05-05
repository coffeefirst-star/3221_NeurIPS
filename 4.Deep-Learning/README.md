# Deep learning and baseline model evaluation

This folder contains the model-evaluation scripts used after the final HTS-derived dataset and split columns have been created. The notebook `splitting_datasets.ipynb` defines the two splitting strategies used here: the Active-NN split and the scaffold split. The model scripts then consume a fixed `split` column from the input CSV rather than creating a new split internally.

The message-passing neural network code is included locally in `BMPNN/` and is imported by `cross_validate_gnn.py` as `BMPNNs.GNNTrainer`.

## Environment setup for BMPNN/PyTorch Geometric

The following CUDA 12.9 / PyTorch 2.8 / PyG environment worked for the GNN evaluation:

```bash
conda create -n pyg_env python=3.11 -y
conda activate pyg_env
conda install nvidia::cuda-toolkit==12.9.0
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```

Install the local BMPNN module from this folder:

```bash
cd /path/to/code/4.Deep-Learning/BMPNN
pip install -e .
```

Then return to `4.Deep-Learning/` before running the model scripts.

## Required input CSV

Each script expects a CSV with at least:

| Column | Description |
|---|---|
| `Title` or `Molecule Name` | Compound identifier. `Title` is used when available. |
| `SMILES` | Molecular structure as a SMILES string. |
| `Label` | Binary class label, where active compounds are `1` and inactive/decoy compounds are `0`. |
| `split` | Precomputed split assignment with values `train`, `val`, or `test`. |

The final packaged dataset contains named split columns such as `Active-NN_split` and `Scaffold_split`, created in `splitting_datasets.ipynb`. The submitted `Final_dataset_tmp_with_stats.csv` is available from Zenodo at https://zenodo.org/records/20030796. To run these scripts, provide a copy of the dataset where the desired split column has been renamed or copied to `split`.

Example for the Active-NN split:

```python
import pandas as pd

df = pd.read_csv("Final_dataset_tmp_with_stats.csv")
df["split"] = df["Active-NN_split"]
df.to_csv("umap_dataset.csv", index=False)
```

Example for the scaffold split:

```python
import pandas as pd

df = pd.read_csv("Final_dataset_tmp_with_stats.csv")
df["split"] = df["Scaffold_split"]
df.to_csv("scaffold_dataset.csv", index=False)
```

## Random forest baseline

Script:

```text
cross_validation_randomforest.py
```

This script converts SMILES to 2048-bit ECFP4/Morgan radius 2 fingerprints and trains a `RandomForestClassifier` using the fixed `train`, `val`, and `test` rows. It selects the tree depth by the smallest train-validation AUPRC gap, then reports validation and blind-test metrics.

Example command:

```bash
python cross_validation_randomforest.py \
  --csv umap_dataset.csv \
  --out-dir rf_umap_not_weighted \
  --split-seed 42 \
  --n-estimators 1000 \
  --max-depth-grid 40
```

Main outputs include:

- `<dataset>_blind_test_predictions.csv`
- `<dataset>_validation_curve.csv`
- `<dataset>_max_depth_selection.csv`
- `<dataset>_validation_curve.png`
- `<dataset>_blind_test_roc.png`
- `<dataset>_blind_test_pr.png`
- `rf_fixed_80_10_10_summary.csv`

## GNN/BMPNN evaluation

Script:

```text
cross_validate_gnn.py
```

This script uses `BMPNNs.GNNTrainer` with the fixed split column from the input CSV. It trains on `train`, monitors `val`, evaluates `test` each epoch, and selects the final blind-test result from the epoch with the best validation AUPRC.

The local `BMPNNs` module must be available on `PYTHONPATH` or installed in the active Python environment. The recommended approach is the editable install shown above from `4.Deep-Learning/BMPNN`.

Example command:

```bash
python cross_validate_gnn.py \
  --csv umap_dataset.csv \
  --input-dir fixed_80_10_10_gnn_output \
  --split-seed 42 \
  --epochs 50 \
  --batch-size 16 \
  --hidden-channels 50 \
  --lr 1e-4 \
  --dropout-rate 0.1
```

Main outputs include:

- `fixed_80_10_10_epoch_metrics.csv`
- `fixed_80_10_10_blind_test_summary.csv`
- `fixed_80_10_10_blind_test_predictions.csv`
- `fixed_80_10_10_val_predictions.csv`
- `fixed_80_10_10_gnn_curves.png`
- ROC and PR curve figures for validation and blind-test sets
- `fixed_80_10_10_gnn.log`

## Recommended evaluation workflow

1. Use `splitting_datasets.ipynb` to define or regenerate the Active-NN and scaffold split assignments.
2. Create one CSV per split strategy by copying the desired split column to `split`.
3. Run the random forest baseline on each split CSV.
4. Install the local `BMPNN` package and run the GNN/BMPNN model on each split CSV.
5. Compare blind-test AUPRC, AUROC, F1, confusion-matrix counts, and train-validation gaps across split strategies.

## Notes

- The scripts assume binary classification.
- Invalid SMILES in the random forest script are converted to all-zero fingerprints; inspect the input CSV before final reporting.
- The GNN script uses fixed random seeds where possible, but GPU operations and package versions may still introduce small numerical differences.
- Outputs are written to the directory passed with `--out-dir` or `--input-dir`.
