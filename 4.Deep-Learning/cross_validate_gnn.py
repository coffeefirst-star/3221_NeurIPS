#!/usr/bin/env python3
import argparse
import inspect
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)



from BMPNNs import GNNTrainer

print(inspect.getfile(GNNTrainer))

INPUT_DIR = "fixed_80_10_10_gnn_output"
DEFAULT_TRAIN_CSV = (
    "umap_dataset.csv"
)
DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BMPNN using a precomputed train/val/test split column from the input CSV."
    )
    parser.add_argument("--csv", default=DEFAULT_TRAIN_CSV, help="CSV file with SMILES, Label, and split columns.")
    parser.add_argument("--input-dir", default=INPUT_DIR, help="Directory for outputs.")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--preprocess-workers", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--curve-plot-out", default="fixed_80_10_10_gnn_curves.png")
    return parser.parse_args()


def pick_name_column(df):
    for col in ("Molecule Name", "Title", "Name", "ID"):
        if col in df.columns:
            return col
    return None


def optional_list(df, column):
    return df[column].tolist() if column in df.columns else None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def subset_list(values, indices):
    if values is None:
        return None
    return [values[idx] for idx in indices]


def make_or_load_split(df, smiles, labels, args, out_dir):
    if "split" not in df.columns:
        raise RuntimeError("Dataset must contain a 'split' column with train/val/test.")

    assignment = df["split"].astype(str).str.strip().str.lower().to_numpy()

    valid = {"train", "val", "test"}
    invalid = sorted(set(assignment) - valid)
    if invalid:
        raise RuntimeError(f"Invalid split labels found: {invalid}")

    counts = {split: int(np.sum(assignment == split)) for split in ("train", "val", "test")}
    if any(count == 0 for count in counts.values()):
        raise RuntimeError(f"Split column must contain non-empty train/val/test splits. Counts: {counts}")

    print("Using precomputed split from CSV (no Butina, no internal splitting).")

    return assignment

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    smiles = df["SMILES"].astype(str).tolist()
    name_col = pick_name_column(df)
    names = df[name_col].astype(str).tolist() if name_col else [f"mol_{idx}" for idx in range(len(df))]
    labels = optional_list(df, "Label")
    if labels is None:
        raise RuntimeError("Fixed-split classification requires a Label column.")
    return df, smiles, names, labels




def metrics_from_probs(targets, probs, preds):
    targets = np.asarray(targets, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = np.asarray(preds, dtype=int)
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    return {
        "auprc": average_precision_score(targets, probs) if np.unique(targets).size > 1 else np.nan,
        "auroc": roc_auc_score(targets, probs) if np.unique(targets).size > 1 else np.nan,
        "f1": f1_score(targets, preds, zero_division=0),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def evaluate_loader(trainer, loader):
    acc, _, _, f1, auprc, loss, targets, probs, names, preds, _ = trainer.evaluate(loader, generate_images=False)
    auroc = roc_auc_score(targets, probs) if np.unique(targets).size > 1 else np.nan
    return {
        "acc": acc,
        "f1": f1,
        "auprc": auprc,
        "auroc": auroc,
        "loss": loss,
        "targets": targets,
        "probs": probs,
        "preds": preds,
        "names": names,
    }


def plot_training_curves(history_df, out_png):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for metric, ax in zip(["loss", "auprc", "auroc", "f1"], axes):
        for split in ("train", "val", "test"):
            sub = history_df[history_df["split"] == split]
            ax.plot(sub["epoch"], sub[metric], label=split)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=800)
    plt.close(fig)


def plot_roc_pr(y_test, probs, out_prefix):
    fpr, tpr, _ = roc_curve(y_test, probs)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC={roc_auc_score(y_test, probs):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_roc.png", dpi=800)
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label=f"AUPRC={average_precision_score(y_test, probs):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pr.png", dpi=800)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.split_seed)
    os.environ["BMPNN_PREPROCESS_WORKERS"] = str(max(1, int(args.preprocess_workers)))
    out_dir = Path(args.input_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(out_dir / "fixed_80_10_10_gnn.log"),
        filemode="w",
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )

    csv_path = Path(args.csv).expanduser().resolve()
    full_df, smiles, names, labels = load_dataset(csv_path)
    labels_arr = np.asarray(labels, dtype=int)
    assignment = make_or_load_split(full_df, smiles, labels_arr, args, out_dir)
    train_idx = np.where(assignment == "train")[0]
    val_idx = np.where(assignment == "val")[0]
    test_idx = np.where(assignment == "test")[0]

    print(f"Using CSV: {csv_path}")
    print(f"Rows train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"Positives train/val/test: {np.sum(labels_arr[train_idx])}/{np.sum(labels_arr[val_idx])}/{np.sum(labels_arr[test_idx])}")

    trainer = GNNTrainer(
        smiles_list=subset_list(smiles, train_idx),
        labels=subset_list(labels, train_idx),
        names_list=subset_list(names, train_idx),
        test_smiles_list=subset_list(smiles, val_idx),
        test_labels=subset_list(labels, val_idx),
        test_names_list=subset_list(names, val_idx),
        node_block="ABMP",
        hidden_channels=args.hidden_channels,
        task="Classification",
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        dropout_rate=args.dropout_rate,
        input_dir=str(out_dir),
        source_file=str(csv_path),
        curve_plot_out=args.curve_plot_out,
    )

    train_dataset = trainer._create_dataset(
        subset_list(smiles, train_idx), subset_list(names, train_idx), subset_list(labels, train_idx), None
    )
    val_dataset = trainer._create_dataset(
        subset_list(smiles, val_idx), subset_list(names, val_idx), subset_list(labels, val_idx), None
    )
    test_dataset = trainer._create_dataset(
        subset_list(smiles, test_idx), subset_list(names, test_idx), subset_list(labels, test_idx), None
    )

    train_loader = trainer._build_train_loader(train_dataset)
    train_eval_loader = trainer._build_eval_loader(train_dataset)
    val_loader = trainer._build_eval_loader(val_dataset)
    test_loader = trainer._build_eval_loader(test_dataset)
    trainer.global_dim = train_dataset.global_dim
    trainer.edge_dim = train_dataset.edge_dim
    trainer.num_node_features = train_dataset.num_node_features
    trainer.setup_model()

    history_rows = []
    best_val_auprc = -np.inf
    best_epoch = None
    best_test_eval = None
    best_val_eval = None

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        trainer.train(train_loader)
        train_eval = evaluate_loader(trainer, train_eval_loader)
        val_eval = evaluate_loader(trainer, val_loader)
        test_eval = evaluate_loader(trainer, test_loader)
        trainer.scheduler.step(val_eval["loss"])

        for split, ev in [("train", train_eval), ("val", val_eval), ("test", test_eval)]:
            history_rows.append(
                {
                    "epoch": epoch,
                    "split": split,
                    "loss": ev["loss"],
                    "auprc": ev["auprc"],
                    "auroc": ev["auroc"],
                    "f1": ev["f1"],
                    "acc": ev["acc"],
                }
            )

        if np.isfinite(val_eval["auprc"]) and val_eval["auprc"] > best_val_auprc:
            best_val_auprc = val_eval["auprc"]
            best_epoch = epoch
            best_test_eval = test_eval
            best_val_eval = val_eval

        print(
            f"Epoch {epoch} | "
            f"Loss train/val/test {train_eval['loss']:.4f}/{val_eval['loss']:.4f}/{test_eval['loss']:.4f} | "
            f"AUPRC {train_eval['auprc']:.4f}/{val_eval['auprc']:.4f}/{test_eval['auprc']:.4f} | "
            f"AUROC {train_eval['auroc']:.4f}/{val_eval['auroc']:.4f}/{test_eval['auroc']:.4f} | "
            f"F1 {train_eval['f1']:.4f}/{val_eval['f1']:.4f}/{test_eval['f1']:.4f} | "
            f"Time {time.time() - start:.2f}s"
        )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(out_dir / "fixed_80_10_10_epoch_metrics.csv", index=False)
    plot_training_curves(history_df, out_dir / args.curve_plot_out)

    if best_test_eval is None:
        raise RuntimeError("No finite validation AUPRC found; cannot select blind-test epoch.")

    test_metrics = metrics_from_probs(best_test_eval["targets"], best_test_eval["probs"], best_test_eval["preds"])
    val_metrics = metrics_from_probs(best_val_eval["targets"], best_val_eval["probs"], best_val_eval["preds"])
    summary = {
        "selection_metric": "best_validation_auprc",
        "best_epoch": int(best_epoch),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "train_positives": int(np.sum(labels_arr[train_idx])),
        "val_positives": int(np.sum(labels_arr[val_idx])),
        "test_positives": int(np.sum(labels_arr[test_idx])),
    }
    summary.update({f"val_{k}": v for k, v in val_metrics.items()})
    summary.update({f"test_{k}": v for k, v in test_metrics.items()})
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(out_dir / "fixed_80_10_10_blind_test_summary.csv", index=False)

    # Save test predictions
    name_to_row = {name: idx for idx, name in enumerate(names)}
    test_pred_names = list(best_test_eval["names"])
    test_pred_df = pd.DataFrame(
        {
            "row_index": [name_to_row.get(name, -1) for name in test_pred_names],
            "name": test_pred_names,
            "y_true": best_test_eval["targets"],
            "y_prob": best_test_eval["probs"],
            "y_pred": best_test_eval["preds"].astype(int),
        }
    )
    test_pred_df.to_csv(out_dir / "fixed_80_10_10_blind_test_predictions.csv", index=False)
    plot_roc_pr(best_test_eval["targets"], best_test_eval["probs"], str(out_dir / "fixed_80_10_10_blind_test"))

    # Save validation predictions
    val_pred_names = list(best_val_eval["names"])
    val_pred_df = pd.DataFrame(
        {
            "row_index": [name_to_row.get(name, -1) for name in val_pred_names],
            "name": val_pred_names,
            "y_true": best_val_eval["targets"],
            "y_prob": best_val_eval["probs"],
            "y_pred": best_val_eval["preds"].astype(int),
        }
    )
    val_pred_df.to_csv(out_dir / "fixed_80_10_10_val_predictions.csv", index=False)
    plot_roc_pr(best_val_eval["targets"], best_val_eval["probs"], str(out_dir / "fixed_80_10_10_val"))

    print(summary_df.to_string(index=False))
    print(f"Saved outputs in: {out_dir}")


if __name__ == "__main__":
    main()
