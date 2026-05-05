#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


DEFAULT_SPLIT_CSV = (
    "umap_dataset.csv"
)
RANDOM_STATE = 42
N_ESTIMATORS = 1000
MAX_DEPTH_GRID = [40]
N_BITS = 2048
RADIUS = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Random forest benchmark using a precomputed train/val/test split CSV."
    )
    parser.add_argument("--csv", default=DEFAULT_SPLIT_CSV, help="CSV file with SMILES, Label, and split columns.")
    parser.add_argument("--out-dir", default="rf_umap_not_weighted")
    parser.add_argument("--split-seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    parser.add_argument("--max-depth-grid", default=",".join(str(n) for n in MAX_DEPTH_GRID))
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def smiles_to_fps(smiles_list, n_bits=N_BITS, radius=RADIUS):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        arr = np.zeros(n_bits, dtype=np.int8)
        if mol is not None:
            fp = generator.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return np.asarray(fps, dtype=np.int8)


def read_split_assignment(df):
    required = {"SMILES", "Label", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Split CSV is missing required columns: {missing}")
    assignment = df["split"].astype(str).str.strip().str.lower().to_numpy()
    valid_splits = {"train", "val", "test"}
    unknown = sorted(set(assignment) - valid_splits)
    if unknown:
        raise RuntimeError(f"Split column contains unsupported values: {unknown}")
    counts = {split: int(np.sum(assignment == split)) for split in ("train", "val", "test")}
    if any(count == 0 for count in counts.values()):
        raise RuntimeError(f"Split column must contain non-empty train/val/test splits. Counts: {counts}")
    return assignment


def save_split_csvs(df, assignment, out_dir, dataset_name):
    split_dir = out_dir / f"{dataset_name}_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        split_df_out = df.loc[assignment == split].copy()
        if "split" in split_df_out.columns:
            split_df_out["split"] = split
        else:
            split_df_out.insert(0, "split", split)
        split_df_out.to_csv(split_dir / f"{dataset_name}_{split}.csv", index=False)


def metrics_from_probs(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "auprc": average_precision_score(y_true, probs) if np.unique(y_true).size > 1 else np.nan,
        "auroc": roc_auc_score(y_true, probs) if np.unique(y_true).size > 1 else np.nan,
        "f1": f1_score(y_true, preds, zero_division=0),
        "tn": int(confusion_matrix(y_true, preds, labels=[0, 1])[0, 0]),
        "fp": int(confusion_matrix(y_true, preds, labels=[0, 1])[0, 1]),
        "fn": int(confusion_matrix(y_true, preds, labels=[0, 1])[1, 0]),
        "tp": int(confusion_matrix(y_true, preds, labels=[0, 1])[1, 1]),
    }


def plot_curves(curve_df, out_png, dataset_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for metric, ax in zip(["auprc", "auroc", "f1"], axes):
        for split in ["train", "val"]:
            sub = curve_df[curve_df["split"] == split]
            ax.plot(sub["max_depth"], sub[metric], marker="o", label=split)
        ax.set_xlabel("max_depth")
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
        ax.legend(frameon=False)
    fig.suptitle(f"{dataset_name}: RF train/validation curve")
    fig.tight_layout()
    fig.savefig(out_png, dpi=800)
    plt.close(fig)


def plot_roc_pr(y_test, test_probs, out_prefix):
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    precision, recall, _ = precision_recall_curve(y_test, test_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC={roc_auc_score(y_test, test_probs):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_roc.png", dpi=800)
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label=f"AUPRC={average_precision_score(y_test, test_probs):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pr.png", dpi=800)
    plt.close()


def run_dataset(df, dataset_name, args, out_dir):
    smiles = df["SMILES"].astype(str).to_numpy()
    y = df["Label"].astype(int).to_numpy()
    X = smiles_to_fps(smiles)
    assignment = read_split_assignment(df)
    save_split_csvs(df, assignment, out_dir, dataset_name)

    train_idx = np.where(assignment == "train")[0]
    val_idx = np.where(assignment == "val")[0]
    test_idx = np.where(assignment == "test")[0]
    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        counts = {split: int(np.sum(assignment == split)) for split in ("train", "val", "test")}
        raise RuntimeError(f"Split assignment must contain non-empty train/val/test splits. Counts: {counts}")

    curve_rows = []
    models = {}
    for max_depth in parse_int_list(args.max_depth_grid):
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            n_jobs=args.n_jobs,
            random_state=args.split_seed,
        )
        model.fit(X[train_idx], y[train_idx])
        models[max_depth] = model
        for split, idx in [("train", train_idx), ("val", val_idx)]:
            probs = model.predict_proba(X[idx])[:, 1]
            row = {"dataset": dataset_name, "split": split, "max_depth": max_depth}
            row.update(metrics_from_probs(y[idx], probs))
            curve_rows.append(row)

    curve_df = pd.DataFrame(curve_rows)
    selection_df = curve_df.pivot(index="max_depth", columns="split", values="auprc").reset_index()
    selection_df["train_val_auprc_gap"] = (selection_df["train"] - selection_df["val"]).abs()
    best_row = selection_df.sort_values(["train_val_auprc_gap", "val"], ascending=[True, False]).iloc[0]
    best_depth = int(best_row["max_depth"])
    best_model = models[best_depth]
    val_probs = best_model.predict_proba(X[val_idx])[:, 1]
    test_probs = best_model.predict_proba(X[test_idx])[:, 1]
    val_metrics = metrics_from_probs(y[val_idx], val_probs)
    test_metrics = metrics_from_probs(y[test_idx], test_probs)

    pred_df = pd.DataFrame(
        {
            "dataset": dataset_name,
            "split": "test",
            "row_index": df["row_index"].to_numpy()[test_idx] if "row_index" in df.columns else test_idx,
            "y_true": y[test_idx],
            "y_prob": test_probs,
            "y_pred": (test_probs >= 0.5).astype(int),
        }
    )
    pred_df.to_csv(out_dir / f"{dataset_name}_blind_test_predictions.csv", index=False)
    curve_df.to_csv(out_dir / f"{dataset_name}_validation_curve.csv", index=False)
    selection_df.to_csv(out_dir / f"{dataset_name}_max_depth_selection.csv", index=False)
    plot_curves(curve_df, out_dir / f"{dataset_name}_validation_curve.png", dataset_name)
    plot_roc_pr(y[test_idx], test_probs, str(out_dir / f"{dataset_name}_blind_test"))

    summary = {
        "dataset": dataset_name,
        "selection_metric": "smallest_train_val_auprc_gap",
        "n_estimators": int(args.n_estimators),
        "best_max_depth_by_train_val_auprc_gap": best_depth,
        "best_train_val_auprc_gap": float(best_row["train_val_auprc_gap"]),
        "selected_train_auprc": float(best_row["train"]),
        "selected_val_auprc": float(best_row["val"]),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "train_positives": int(np.sum(y[train_idx] == 1)),
        "val_positives": int(np.sum(y[val_idx] == 1)),
        "test_positives": int(np.sum(y[test_idx] == 1)),
    }
    summary.update({f"val_{k}": v for k, v in val_metrics.items()})
    summary.update({f"blind_test_{k}": v for k, v in test_metrics.items()})
    return summary


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = Path(args.csv).expanduser().resolve()
    df = pd.read_csv(split_path)
    summaries = [run_dataset(df, split_path.stem, args=args, out_dir=out_dir)]

    summary_df = pd.DataFrame(summaries)
    summary_path = out_dir / "rf_fixed_80_10_10_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
