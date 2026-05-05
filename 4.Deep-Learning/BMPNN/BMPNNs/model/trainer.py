
from .interaction_network import InteractionNetwork
import os
import csv
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.utils.data import WeightedRandomSampler
from BMPNNs.data.molecular_dataset import MolecularDataset
import time
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score, average_precision_score, precision_score, recall_score, f1_score,
    roc_curve, auc, root_mean_squared_error, matthews_corrcoef,
    confusion_matrix
)
from sklearn.model_selection import KFold, StratifiedKFold
from matplotlib.colors import LinearSegmentedColormap
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from collections import defaultdict
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import random


seed = 122
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")


class GNNTrainer:
    def __init__(self, smiles_list, labels=None, names_list=None, pEC50_labels=None,
                 node_block="BMP", hidden_channels=64, task="Classification",
                 num_node_features=6, global_dim=6, lr=0.001, edge_dim=4,
                 batch_size=32, k_folds=5, dropout_rate=0.5,
                 input_dir="evaluate_outputs", max_norm=1, epochs=50,
                 threshold=0.5, num_classes=2, multitask_loss_weights=None,
                 activity_threshold=5.0, scheduler_patience=10, source_file=None,
                 test_smiles_list=None, test_labels=None, test_names_list=None,
                 test_pEC50_labels=None, curve_plot_out="cross_validation_gnn.png"):
        """
        Args:
            smiles_list: List of SMILES strings
            labels: Binary classification labels (0/1) - can be None for test
            names_list: Compound names
            pEC50_labels: Regression labels (pEC50 values) - required for MultiTask
            node_block: GNN variant (BMP, ABMP, CBMP, etc.)
            task: "Classification", "Regression", or "MultiTask"
            activity_threshold: pEC50 threshold for deriving classification labels
            multitask_loss_weights: dict with "cls" and "active" keys for loss weighting
        """
        self.smiles_list = smiles_list
        self.labels = labels
        self.names_list = names_list
        self.pEC50_labels = pEC50_labels
        self.hidden_channels = hidden_channels
        self.num_node_features = num_node_features
        self.global_dim = global_dim
        self.lr = lr
        self.task = task
        self.batch_size = batch_size
        self.k_folds = k_folds
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout_rate = dropout_rate
        self.edge_dim = edge_dim
        self.threshold = threshold
        self.input_dir = input_dir
        self.node_block = node_block
        self.max_norm = max_norm
        self.num_classes = num_classes
        self.multitask_loss_weights = multitask_loss_weights or {"cls": 1.0, "active": 1.0}
        self.activity_threshold = activity_threshold
        self.scheduler_patience = scheduler_patience
        self.source_file = source_file
        self.test_smiles_list = test_smiles_list
        self.test_labels = test_labels
        self.test_names_list = test_names_list
        self.test_pEC50_labels = test_pEC50_labels
        self.curve_plot_out = curve_plot_out

        if self.task == "Classification" and self.labels is None:
            raise ValueError("Classification mode requires Label values.")
        if self.task == "Regression" and self.pEC50_labels is None:
            raise ValueError("Regression mode requires pEC50 values.")
        if self.task == "MultiTask":
            if self.labels is None:
                raise ValueError("MultiTask mode requires Label values.")
            if self.pEC50_labels is None:
                raise ValueError("MultiTask mode requires pEC50 values.")

        if self.test_smiles_list is not None:
            if self.task == "Classification" and self.test_labels is None:
                raise ValueError("Classification test evaluation requires Label values in the test set.")
            if self.task == "Regression" and self.test_pEC50_labels is None:
                raise ValueError("Regression test evaluation requires pEC50 values in the test set.")
            if self.task == "MultiTask":
                if self.test_labels is None:
                    raise ValueError("MultiTask test evaluation requires Label values in the test set.")
                if self.test_pEC50_labels is None:
                    raise ValueError("MultiTask test evaluation requires pEC50 values in the test set.")

    def _relative_absolute_error(self, targets, predictions, baseline_mean=None):
        targets = np.asarray(targets, dtype=float)
        predictions = np.asarray(predictions, dtype=float)
        if targets.size == 0:
            return np.nan
        if baseline_mean is None:
            baseline_mean = float(targets.mean())
        denom = np.abs(targets - baseline_mean).sum()
        if denom == 0:
            return np.nan
        return np.abs(targets - predictions).sum() / denom

    def _compute_regression_metrics(self, targets, predictions, baseline_mean=None):
        targets = np.asarray(targets, dtype=float)
        predictions = np.asarray(predictions, dtype=float)
        if targets.size == 0:
            return {
                "rae": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "baseline_mean": np.nan,
                "baseline_rmse": np.nan,
                "baseline_mae": np.nan,
            }

        if baseline_mean is None:
            baseline_mean = float(targets.mean())

        baseline_predictions = np.full_like(targets, baseline_mean, dtype=float)
        abs_error = np.abs(targets - predictions)

        return {
            "rae": self._relative_absolute_error(targets, predictions, baseline_mean),
            "rmse": root_mean_squared_error(targets, predictions),
            "mae": float(abs_error.mean()),
            "baseline_mean": float(baseline_mean),
            "baseline_rmse": root_mean_squared_error(targets, baseline_predictions),
            "baseline_mae": float(np.abs(targets - baseline_predictions).mean()),
        }

    def _build_train_loader(self, train_dataset):
        train_labels = np.asarray(getattr(train_dataset, "successful_labels", []), dtype=int)
        use_weighted_sampler = (
            self.task in {"Classification", "MultiTask"}
            and train_labels.size > 0
            and np.unique(train_labels).size > 1
        )
        use_weighted_sampler = False

        if use_weighted_sampler:
            class_weights = 1.0 / np.bincount(train_labels)
            sample_weights = class_weights[train_labels]
            sampler = WeightedRandomSampler(
                torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            print(
                "Using weighted sampler: "
                f"class_weights={class_weights.tolist()} "
                f"train_pos={int(train_labels.sum())} "
                f"train_neg={int(train_labels.size - train_labels.sum())}"
            )
            return GeometricDataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                shuffle=False,
                drop_last=True,
            )

        return GeometricDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,

        )

    def setup_model(self):
        self.model = InteractionNetwork(
            self.num_node_features,
            self.edge_dim,
            self.hidden_channels,
            self.global_dim,
            self.dropout_rate,
            self.node_block,
            task=self.task,
            num_classes=self.num_classes,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.task == "Regression":
            self.criterion = nn.MSELoss()
        elif self.task == "Classification":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.task == "MultiTask":
            self.classification_criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported task type: {self.task}")

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=self.scheduler_patience
        )

        print(f"\nMode: {self.node_block.upper() if isinstance(self.node_block, str) else type(self.node_block).__name__}")
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Total trainable parameters: {total_params:,}\n")

    def _compute_regression_loss_from_batch(self, out, data):
        targets = data.y_active.view(-1).float()
        preds = out.view(-1)
        mask = data.mask_active.view(-1) > 0
        if not mask.any():
            return None
        return self.criterion(preds[mask], targets[mask])

    def _compute_multitask_loss(self, outputs, data):
        """Compute MultiTask loss with proper masking for regression"""
        class_logits = outputs["class_logits"].view(-1)
        active_pred = outputs["active_pred"].view(-1)

        y_cls = data.y_cls.view(-1).float()
        y_active = data.y_active.view(-1).float()
        mask_active = data.mask_active.view(-1) > 0

        # Classification loss
        loss_cls = self.classification_criterion(class_logits, y_cls)

        # Regression loss (only where mask_active=1)
        if mask_active.any():
            pred_reg = active_pred[mask_active]
            target_reg = y_active[mask_active]
            # Use MSE for regression
            loss_active = nn.MSELoss()(pred_reg, target_reg)
        else:
            loss_active = torch.zeros((), device=active_pred.device)

        # Weighted combination
        cls_weight = self.multitask_loss_weights.get("cls", 1.0)
        active_weight = self.multitask_loss_weights.get("active", 1.0)
        total = cls_weight * loss_cls + active_weight * loss_active

        return total, loss_cls.detach(), loss_active.detach()

    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        total_weight = 0
        all_names = []
        all_labels = []
        all_predictions = []
        class_targets = []
        class_predictions = []
        active_targets = []
        active_predictions = []

        for data in train_loader:
            data = data.to(self.device)

            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.edge_attr = data.edge_attr.to(device)
            data.u = data.u.to(device)
            data.batch = data.batch.to(device)

            if self.task != "MultiTask":
                data.y = data.y.view(-1, 1)

            self.optimizer.zero_grad()

            out, _ = self.model(
                data.x, data.edge_index, data.edge_attr, data.u, data.batch
            )

            if self.task == "MultiTask":
                loss, _, _ = self._compute_multitask_loss(out, data)
            elif self.task == "Regression":
                loss = self._compute_regression_loss_from_batch(out, data)
                if loss is None:
                    continue
            else:
                loss = self.criterion(out, data.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()

            if self.task == "Regression":
                batch_weight = int((data.mask_active.view(-1) > 0).sum().item())
            else:
                batch_weight = data.num_graphs
            total_loss += loss.item() * batch_weight
            total_weight += batch_weight
            all_names.extend(data.name)

            if self.task == "MultiTask":
                class_logits = out["class_logits"]
                pred_cls = (torch.sigmoid(class_logits.view(-1)) > self.threshold).long()
                class_targets.extend(data.y_cls.detach().cpu().numpy().flatten())
                class_predictions.extend(pred_cls.detach().cpu().numpy().flatten())

                active_mask = data.mask_active.view(-1) > 0
                if active_mask.any():
                    active_targets.extend(
                        data.y_active.view(-1)[active_mask].detach().cpu().numpy().flatten()
                    )
                    active_predictions.extend(
                        out["active_pred"].view(-1)[active_mask].detach().cpu().numpy().flatten()
                    )
            else:
                if self.task == "Regression":
                    active_mask = data.mask_active.view(-1) > 0
                    if active_mask.any():
                        all_labels.append(data.y_active.view(-1)[active_mask].cpu().numpy())
                        all_predictions.extend(out.view(-1)[active_mask].detach().cpu().numpy().flatten())
                else:
                    probabilities = torch.sigmoid(out)
                    pred = (probabilities > self.threshold).float()
                    all_labels.append(data.y.cpu().numpy())
                    all_predictions.extend(pred.detach().cpu().numpy().flatten())

        if self.task != "MultiTask":
            self.all_labels = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([])
            self.all_names = all_names
            self.all_predictions = np.array(all_predictions)
        else:
            self.train_multitask_summary = {
                "class_targets": np.asarray(class_targets),
                "class_predictions": np.asarray(class_predictions),
                "active_targets": np.asarray(active_targets),
                "active_predictions": np.asarray(active_predictions),
            }

        return total_loss / max(1, total_weight)

    def _create_multitask_dataset(self, smiles, names, labels, pEC50_vals):
        """Create dataset with MultiTask support"""
        return MolecularDataset(
            smiles_list=smiles,
            names_list=names,
            labels=labels,
            pEC50_labels=pEC50_vals,
            activity_threshold=self.activity_threshold,
            node_block=self.node_block,
            source_file=self.source_file,
        )

    def _create_dataset(self, smiles, names, labels, pEC50_vals):
        if self.task == "MultiTask":
            return self._create_multitask_dataset(smiles, names, labels, pEC50_vals)
        if self.task == "Regression":
            return MolecularDataset(
                smiles_list=smiles,
                names_list=names,
                labels=labels,
                pEC50_labels=pEC50_vals,
                activity_threshold=self.activity_threshold,
                node_block=self.node_block,
                source_file=self.source_file,
            )
        return MolecularDataset(
            smiles, names, labels, node_block=self.node_block, source_file=self.source_file
        )

    def _build_eval_loader(self, dataset):
        return GeometricDataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,

        )

    def _plot_cv_loss_curves(self, train_losses, val_losses, test_losses=None):
        if len(train_losses) == 0:
            return

        train_arr = np.asarray(train_losses, dtype=float)
        val_arr = np.asarray(val_losses, dtype=float)
        test_arr = np.asarray(test_losses, dtype=float) if test_losses else None

        epochs = np.arange(1, train_arr.shape[1] + 1)
        train_mean = train_arr.mean(axis=0)
        val_mean = val_arr.mean(axis=0)
        train_std = train_arr.std(axis=0, ddof=1) if train_arr.shape[0] > 1 else np.zeros_like(train_mean)
        val_std = val_arr.std(axis=0, ddof=1) if val_arr.shape[0] > 1 else np.zeros_like(val_mean)
        best_epoch = int(np.argmin(val_mean)) + 1

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_mean, label="train (mean)")
        plt.plot(epochs, val_mean, label="validation (mean)")
        plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2)
        plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2)

        if test_arr is not None and test_arr.size > 0:
            test_mean = test_arr.mean(axis=0)
            test_std = test_arr.std(axis=0, ddof=1) if test_arr.shape[0] > 1 else np.zeros_like(test_mean)
            plt.plot(epochs, test_mean, label="test (mean)")
            plt.fill_between(epochs, test_mean - test_std, test_mean + test_std, alpha=0.2)

        plt.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"best val epoch = {best_epoch}")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"{self.node_block}: Train vs Validation vs Test Loss per Epoch ({self.task})")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(self.input_dir, self.curve_plot_out)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved CV loss curve to {plot_path}")

    def _standard_error(self, values):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size <= 1:
            return 0.0
        return float(np.std(arr, ddof=1) / np.sqrt(arr.size))

    def cross_validate(self):
        if self.task in {"Classification", "MultiTask"} and self.labels is not None:
            kf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=seed)
            split_iterator = kf.split(self.smiles_list, self.labels)
        else:
            kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=seed)
            split_iterator = kf.split(self.smiles_list)

        fold_val_losses = []
        fold_train_losses = []
        fold_test_losses = []

        if self.task == "MultiTask":
            fold_train_cls_f1s, fold_val_cls_f1s = [], []
            fold_train_reg_raes, fold_val_reg_raes = [], []
            fold_train_reg_maes, fold_val_reg_maes = [], []
            fold_val_reg_baseline_maes, fold_val_reg_baseline_rmses = [], []
        elif self.task == "Regression":
            fold_train_rmses, fold_val_rmses = [], []
            fold_train_raes, fold_val_raes = [], []
            fold_train_maes, fold_val_maes = [], []
            fold_val_baseline_maes, fold_val_baseline_rmses = [], []
        elif self.task == "Classification":
            fold_train_f1_history, fold_val_f1_history = [], []
            fold_train_auprc_history, fold_val_auprc_history = [], []
        fold_test_primary_metrics = []

        for fold, (train_idx, val_idx) in enumerate(split_iterator):
            print(f"\n{'='*50}\nFold {fold + 1}/{self.k_folds}\n{'='*50}")

            # Split data
            smiles_train = [self.smiles_list[i] for i in train_idx]
            names_train = [self.names_list[i] for i in train_idx]
            labels_train = [self.labels[i] for i in train_idx]

            smiles_val = [self.smiles_list[i] for i in val_idx]
            names_val = [self.names_list[i] for i in val_idx]
            labels_val = [self.labels[i] for i in val_idx]

            if self.task in {"Classification", "MultiTask"}:
                train_pos = int(np.sum(labels_train))
                val_pos = int(np.sum(labels_val))
                print(
                    f"Before filtering: "
                    f"train pos={train_pos} neg={len(labels_train) - train_pos} | "
                    f"val pos={val_pos} neg={len(labels_val) - val_pos}"
                )

            # For MultiTask, also split pEC50 labels
            if self.pEC50_labels is not None:
                pEC50_train = [self.pEC50_labels[i] for i in train_idx]
                pEC50_val = [self.pEC50_labels[i] for i in val_idx]
            else:
                pEC50_train = None
                pEC50_val = None
            train_reg_baseline_mean = None
            if pEC50_train is not None:
                train_reg_values = [float(v) for v in pEC50_train if v is not None and not np.isnan(v)]
                if train_reg_values:
                    train_reg_baseline_mean = float(np.mean(train_reg_values))

            train_dataset = self._create_dataset(
                smiles_train, names_train, labels_train, pEC50_train
            )
            val_dataset = self._create_dataset(
                smiles_val, names_val, labels_val, pEC50_val
            )

            test_loader = None
            if self.test_smiles_list is not None and self.test_names_list is not None:
                test_dataset = self._create_dataset(
                    self.test_smiles_list,
                    self.test_names_list,
                    self.test_labels,
                    self.test_pEC50_labels,
                )
                test_loader = self._build_eval_loader(test_dataset)

            if self.task in {"Classification", "MultiTask"}:
                train_successful = np.asarray(train_dataset.successful_labels)
                val_successful = np.asarray(val_dataset.successful_labels)
                train_successful_pos = int(np.sum(train_successful)) if train_successful.size else 0
                val_successful_pos = int(np.sum(val_successful)) if val_successful.size else 0
                print(
                    f"After filtering: "
                    f"train pos={train_successful_pos} neg={len(train_successful) - train_successful_pos} | "
                    f"val pos={val_successful_pos} neg={len(val_successful) - val_successful_pos}"
                )

            self.global_dim = train_dataset.global_dim
            self.edge_dim = train_dataset.edge_dim
            self.num_node_features = train_dataset.num_node_features

            train_loader = self._build_train_loader(train_dataset)
            val_loader = self._build_eval_loader(val_dataset)

            self.setup_model()
            epoch_train_losses = []
            epoch_val_losses = []
            epoch_test_losses = []

            for epoch in range(self.epochs):
                start_time = time.time()
                _ = self.train(train_loader)

                if self.task == "MultiTask":
                    train_metrics = self.evaluate(
                        train_loader,
                        generate_images=False,
                        regression_baseline_mean=train_reg_baseline_mean,
                    )
                    val_metrics = self.evaluate(
                        val_loader,
                        generate_images=False,
                        regression_baseline_mean=train_reg_baseline_mean,
                    )

                    train_loss = train_metrics["loss"]
                    val_loss = val_metrics["loss"]
                    epoch_train_losses.append(train_loss)
                    epoch_val_losses.append(val_loss)
                    train_cls_f1 = train_metrics["class_f1"]
                    val_cls_f1 = val_metrics["class_f1"]
                    train_reg_rae = train_metrics["reg_rae"]
                    val_reg_rae = val_metrics["reg_rae"]
                    train_reg_mae = train_metrics["reg_mae"]
                    val_reg_mae = val_metrics["reg_mae"]

                    self.scheduler.step(val_loss)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    test_fragment = ""
                    if test_loader is not None:
                        test_metrics = self.evaluate(
                            test_loader,
                            generate_images=False,
                            regression_baseline_mean=train_reg_baseline_mean,
                        )
                        test_loss = test_metrics["loss"]
                        test_cls_f1 = test_metrics["class_f1"]
                        epoch_test_losses.append(test_loss)
                        test_fragment = f" | Test Loss/F1: {test_loss:.4f}/{test_cls_f1:.4f}"

                    print(
                        f"Epoch {epoch+1} | Loss: {train_loss:.4f}/{val_loss:.4f} | "
                        f"Cls F1: {train_cls_f1:.4f}/{val_cls_f1:.4f} | "
                        f"Reg MAE: {train_reg_mae:.4f}/{val_reg_mae:.4f} | "
                        f"Reg RAE: {train_reg_rae:.4f}/{val_reg_rae:.4f} | "
                        f"Baseline RMSE/MAE (val): "
                        f"{val_metrics['reg_baseline_rmse']:.4f}/{val_metrics['reg_baseline_mae']:.4f} | "
                        f"LR: {current_lr:.6f}"
                        f"{test_fragment}"
                    )

                    if epoch == self.epochs - 1:
                        fold_train_cls_f1s.append(train_cls_f1)
                        fold_val_cls_f1s.append(val_cls_f1)
                        fold_train_reg_raes.append(train_reg_rae)
                        fold_val_reg_raes.append(val_reg_rae)
                        fold_train_reg_maes.append(train_reg_mae)
                        fold_val_reg_maes.append(val_reg_mae)
                        fold_val_reg_baseline_maes.append(val_metrics["reg_baseline_mae"])
                        fold_val_reg_baseline_rmses.append(val_metrics["reg_baseline_rmse"])
                        if test_loader is not None:
                            fold_test_primary_metrics.append(test_cls_f1)

                elif self.task == "Classification":
                    train_acc, _, _, train_f1, train_auprc, train_loss, _, _, _, _, _ = self.evaluate(
                        train_loader, generate_images=True
                    )
                    val_acc, _, _, val_f1, val_auprc, val_loss, _, _, _, _, _ = self.evaluate(
                        val_loader, generate_images=False
                    )
                    epoch_train_losses.append(train_loss)
                    epoch_val_losses.append(val_loss)
                    fold_train_f1_history.append(train_f1)
                    fold_val_f1_history.append(val_f1)
                    fold_train_auprc_history.append(train_auprc)
                    fold_val_auprc_history.append(val_auprc)
                    self.scheduler.step(val_loss)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    test_fragment = ""
                    if test_loader is not None:
                        test_acc, _, _, test_f1, test_auprc, test_loss, _, _, _, _, _ = self.evaluate(
                            test_loader, generate_images=False
                        )
                        epoch_test_losses.append(test_loss)
                        test_fragment = (
                            f" | Test Acc/F1/AUPRC/Loss: "
                            f"{test_acc:.4f}/{test_f1:.4f}/{test_auprc:.4f}/{test_loss:.4f}"
                        )
                        if epoch == self.epochs - 1:
                            fold_test_primary_metrics.append(test_f1)

                    print(
                        f"Epoch {epoch+1} | Loss: {train_loss:.4f}/{val_loss:.4f} | "
                        f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
                        f"F1: {train_f1:.4f}/{val_f1:.4f} | "
                        f"AUPRC: {train_auprc:.4f}/{val_auprc:.4f} | "
                        f"Time: {time.time()-start_time:.2f}s | LR: {current_lr:.6f}"
                        f"{test_fragment}"
                    )
                elif self.task == "Regression":
                    train_rmse, train_rae, train_mae, train_baseline_rmse, train_baseline_mae, train_loss, _, _, _ = self.evaluate(
                        train_loader,
                        generate_images=False,
                        regression_baseline_mean=train_reg_baseline_mean,
                    )
                    val_rmse, val_rae, val_mae, val_baseline_rmse, val_baseline_mae, val_loss, _, _, _ = self.evaluate(
                        val_loader,
                        generate_images=False,
                        regression_baseline_mean=train_reg_baseline_mean,
                    )
                    epoch_train_losses.append(train_loss)
                    epoch_val_losses.append(val_loss)
                    self.scheduler.step(val_loss)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    test_fragment = ""
                    if test_loader is not None:
                        test_rmse, test_rae, test_mae, _, _, test_loss, _, _, _ = self.evaluate(
                            test_loader,
                            generate_images=False,
                            regression_baseline_mean=train_reg_baseline_mean,
                        )
                        epoch_test_losses.append(test_loss)
                        test_fragment = (
                            f" | Test RMSE/MAE/RAE/Loss: "
                            f"{test_rmse:.4f}/{test_mae:.4f}/{test_rae:.4f}/{test_loss:.4f}"
                        )

                    print(
                        f"Epoch {epoch+1} | Loss: {train_loss:.4f}/{val_loss:.4f} | "
                        f"RMSE: {train_rmse:.4f}/{val_rmse:.4f} | "
                        f"MAE: {train_mae:.4f}/{val_mae:.4f} | "
                        f"RAE: {train_rae:.4f}/{val_rae:.4f} | "
                        f"Baseline RMSE/MAE (val): {val_baseline_rmse:.4f}/{val_baseline_mae:.4f} | "
                        f"Time: {time.time()-start_time:.2f}s | LR: {current_lr:.6f}"
                        f"{test_fragment}"
                    )

                    if epoch == self.epochs - 1:
                        fold_train_rmses.append(train_rmse)
                        fold_val_rmses.append(val_rmse)
                        fold_train_raes.append(train_rae)
                        fold_val_raes.append(val_rae)
                        fold_train_maes.append(train_mae)
                        fold_val_maes.append(val_mae)
                        fold_val_baseline_maes.append(val_baseline_mae)
                        fold_val_baseline_rmses.append(val_baseline_rmse)
                        if test_loader is not None:
                            fold_test_primary_metrics.append(test_rmse)

            fold_train_losses.append(epoch_train_losses)
            fold_val_losses.append(epoch_val_losses)
            if test_loader is not None:
                fold_test_losses.append(epoch_test_losses)

        # Final summary
        self._plot_cv_loss_curves(
            fold_train_losses,
            fold_val_losses,
            fold_test_losses if len(fold_test_losses) > 0 else None,
        )
        if self.task == "MultiTask":
            print(f"\n{'='*50}")
            print("CROSS-VALIDATION SUMMARY (MultiTask)")
            print(f"{'='*50}")
            print(f"Average Val Cls F1: {np.mean(fold_val_cls_f1s):.4f} +/- {np.std(fold_val_cls_f1s):.4f}")
            print(f"Average Val Reg MAE: {np.mean(fold_val_reg_maes):.4f} +/- {np.std(fold_val_reg_maes):.4f}")
            print(f"Average Val Reg RAE: {np.mean(fold_val_reg_raes):.4f} +/- {np.std(fold_val_reg_raes):.4f}")
            print(f"Average Val Reg Baseline MAE: {np.mean(fold_val_reg_baseline_maes):.4f} +/- {np.std(fold_val_reg_baseline_maes):.4f}")
            print(f"Average Val Reg Baseline RMSE: {np.mean(fold_val_reg_baseline_rmses):.4f} +/- {np.std(fold_val_reg_baseline_rmses):.4f}")
            if len(fold_test_primary_metrics) > 0:
                print(f"Average Test Cls F1: {np.mean(fold_test_primary_metrics):.4f} +/- {np.std(fold_test_primary_metrics):.4f}")
        elif self.task == "Regression":
            print(f"\n{'='*50}")
            print("CROSS-VALIDATION SUMMARY (Regression)")
            print(f"{'='*50}")
            print(f"Average Val RMSE: {np.mean(fold_val_rmses):.4f} +/- {np.std(fold_val_rmses):.4f}")
            print(f"Average Val MAE: {np.mean(fold_val_maes):.4f} +/- {np.std(fold_val_maes):.4f}")
            print(f"Average Val RAE: {np.mean(fold_val_raes):.4f} +/- {np.std(fold_val_raes):.4f}")
            print(f"Average Val Baseline MAE: {np.mean(fold_val_baseline_maes):.4f} +/- {np.std(fold_val_baseline_maes):.4f}")
            print(f"Average Val Baseline RMSE: {np.mean(fold_val_baseline_rmses):.4f} +/- {np.std(fold_val_baseline_rmses):.4f}")
            if len(fold_test_primary_metrics) > 0:
                print(f"Average Test RMSE: {np.mean(fold_test_primary_metrics):.4f} +/- {np.std(fold_test_primary_metrics):.4f}")
        elif self.task == "Classification":
            last_n = min(10, self.epochs)
            n_folds = self.k_folds
            train_f1_last_n = np.asarray(fold_train_f1_history, dtype=float).reshape(n_folds, self.epochs)[:, -last_n:].mean(axis=1)
            val_f1_last_n = np.asarray(fold_val_f1_history, dtype=float).reshape(n_folds, self.epochs)[:, -last_n:].mean(axis=1)
            train_auprc_last_n = np.asarray(fold_train_auprc_history, dtype=float).reshape(n_folds, self.epochs)[:, -last_n:].mean(axis=1)
            val_auprc_last_n = np.asarray(fold_val_auprc_history, dtype=float).reshape(n_folds, self.epochs)[:, -last_n:].mean(axis=1)
            print(f"\n{'='*50}")
            print(f"CROSS-VALIDATION SUMMARY (Classification, last {last_n} epochs)")
            print(f"{'='*50}")
            print(f"Average Train F1: {np.mean(train_f1_last_n):.4f} +/- {self._standard_error(train_f1_last_n):.4f} (SE)")
            print(f"Average Val F1: {np.mean(val_f1_last_n):.4f} +/- {self._standard_error(val_f1_last_n):.4f} (SE)")
            print(f"Average Train AUPRC: {np.mean(train_auprc_last_n):.4f} +/- {self._standard_error(train_auprc_last_n):.4f} (SE)")
            print(f"Average Val AUPRC: {np.mean(val_auprc_last_n):.4f} +/- {self._standard_error(val_auprc_last_n):.4f} (SE)")
            if len(fold_test_primary_metrics) > 0:
                print(f"Average Test F1: {np.mean(fold_test_primary_metrics):.4f} +/- {self._standard_error(fold_test_primary_metrics):.4f} (SE)")
        elif len(fold_test_primary_metrics) > 0:
            print(f"\nAverage Test F1: {np.mean(fold_test_primary_metrics):.4f} +/- {np.std(fold_test_primary_metrics):.4f}")

    def evaluate(self, loader, generate_images=True, regression_baseline_mean=None):
        """Evaluate model - returns metrics dict for MultiTask."""
        self.model.eval()
        total_loss = 0.0
        all_class_targets, all_class_preds = [], []
        all_reg_targets, all_reg_preds = [], []
        reg_mean = getattr(loader.dataset, "pEC50_mean", None)
        reg_std = getattr(loader.dataset, "pEC50_std", None)

        for data in loader:
            data = data.to(self.device)

            with torch.no_grad():
                out, _ = self.model(
                    data.x, data.edge_index, data.edge_attr, data.u, data.batch
                )

                if self.task == "MultiTask":
                    loss, _, _ = self._compute_multitask_loss(out, data)
                    total_loss += loss.item() * data.num_graphs

                    y_cls = data.y_cls.view(-1).float()
                    class_probs = torch.sigmoid(out["class_logits"].view(-1))
                    class_preds = (class_probs > self.threshold).long()

                    all_class_targets.extend(y_cls.detach().cpu().numpy().flatten())
                    all_class_preds.extend(class_preds.detach().cpu().numpy().flatten())

                    mask_active = data.mask_active.view(-1) > 0
                    if mask_active.any():
                        reg_targets = data.y_active.view(-1)[mask_active].detach().cpu().numpy().flatten()
                        reg_preds = out["active_pred"].view(-1)[mask_active].detach().cpu().numpy().flatten()

                        if reg_mean is not None and reg_std is not None:
                            reg_targets = (reg_targets * reg_std) + reg_mean
                            reg_preds = (reg_preds * reg_std) + reg_mean

                        all_reg_targets.extend(reg_targets)
                        all_reg_preds.extend(reg_preds)

        # Compute final metrics
        if self.task == "MultiTask":
            class_targets_arr = np.asarray(all_class_targets)
            class_preds_arr = np.asarray(all_class_preds)
            class_f1 = f1_score(class_targets_arr, class_preds_arr, zero_division=0)
            cm = confusion_matrix(class_targets_arr, class_preds_arr, labels=[0, 1])

            if len(all_reg_targets) > 0:
                reg_targets_arr = np.asarray(all_reg_targets)
                reg_preds_arr = np.asarray(all_reg_preds)
                reg_metrics = self._compute_regression_metrics(
                    reg_targets_arr,
                    reg_preds_arr,
                    baseline_mean=regression_baseline_mean,
                )
                reg_rae = reg_metrics["rae"]
                reg_abs_rae = float(np.abs(reg_targets_arr - reg_preds_arr).sum())
                reg_rmse = reg_metrics["rmse"]
                reg_mae = reg_metrics["mae"]
                reg_baseline_rmse = reg_metrics["baseline_rmse"]
                reg_baseline_mae = reg_metrics["baseline_mae"]
            else:
                reg_targets_arr = np.asarray(all_reg_targets)
                reg_preds_arr = np.asarray(all_reg_preds)
                reg_rae = np.nan
                reg_abs_rae = np.nan
                reg_rmse = np.nan
                reg_mae = np.nan
                reg_baseline_rmse = np.nan
                reg_baseline_mae = np.nan

            return {
                "loss": total_loss / len(loader.dataset),
                "class_f1": class_f1,
                "confusion_matrix": cm,
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
                "reg_rae": reg_rae,
                "reg_abs_rae": reg_abs_rae,
                "reg_rmse": reg_rmse,
                "reg_mae": reg_mae,
                "reg_baseline_rmse": reg_baseline_rmse,
                "reg_baseline_mae": reg_baseline_mae,
                "class_targets": class_targets_arr,
                "class_preds": class_preds_arr,
                "reg_targets": reg_targets_arr,
                "reg_preds": reg_preds_arr,
            }
        elif self.task == "Regression":
            total_loss = 0.0
            all_targets, all_preds = [], []
            all_names = []

            for data in loader:
                data = data.to(self.device)
                with torch.no_grad():
                    out, _ = self.model(
                        data.x, data.edge_index, data.edge_attr, data.u, data.batch
                    )
                    loss = self._compute_regression_loss_from_batch(out, data)
                    if loss is None:
                        continue
                    mask = data.mask_active.view(-1) > 0
                    total_loss += loss.item() * int(mask.sum().item())

                targets = data.y_active.view(-1)[mask].detach().cpu().numpy().flatten()
                preds = out.view(-1)[mask].detach().cpu().numpy().flatten()

                if reg_mean is not None and reg_std is not None:
                    targets = (targets * reg_std) + reg_mean
                    preds = (preds * reg_std) + reg_mean

                all_targets.extend(targets)
                all_preds.extend(preds)
                all_names.extend([name for idx, name in enumerate(data.name) if bool(mask[idx].item())])

            metrics = self._compute_regression_metrics(
                all_targets,
                all_preds,
                baseline_mean=regression_baseline_mean,
            )
            return (
                metrics["rmse"],
                metrics["rae"],
                metrics["mae"],
                metrics["baseline_rmse"],
                metrics["baseline_mae"],
                total_loss / max(1, len(all_targets)),
                np.asarray(all_targets),
                np.asarray(all_preds),
                all_names,
            )
        else:
            total_loss = 0.0
            all_targets, all_probs, all_preds = [], [], []
            all_names = []

            for data in loader:
                data = data.to(self.device)
                with torch.no_grad():
                    out, _ = self.model(
                        data.x, data.edge_index, data.edge_attr, data.u, data.batch
                    )
                    loss = self.criterion(out.view(-1), data.y.view(-1).float())
                    total_loss += loss.item() * data.num_graphs

                probs = torch.sigmoid(out.view(-1))
                preds_bin = (probs > self.threshold).float()
                all_targets.extend(data.y.view(-1).detach().cpu().numpy().flatten())
                all_probs.extend(probs.detach().cpu().numpy().flatten())
                all_preds.extend(preds_bin.detach().cpu().numpy().flatten())
                all_names.extend(data.name)

            accuracy = accuracy_score(all_targets, all_preds)
            precision = precision_score(all_targets, all_preds, zero_division=0)
            recall = recall_score(all_targets, all_preds, zero_division=0)
            f1 = f1_score(all_targets, all_preds, zero_division=0)
            auprc = average_precision_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else np.nan
            cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
            print(
                f"  Eval debug: n={len(all_targets)} "
                f"true_pos={int(np.sum(all_targets))} "
                f"pred_pos={int(np.sum(all_preds))}"
            )

            return (
                accuracy,
                precision,
                recall,
                f1,
                auprc,
                total_loss / len(loader.dataset),
                np.asarray(all_targets),
                np.asarray(all_probs),
                all_names,
                np.asarray(all_preds),
                cm,
            )
