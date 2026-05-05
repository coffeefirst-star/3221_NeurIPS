
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader as GeometricDataLoader
from BMPNNs import GNNTrainer
from BMPNNs.data.molecular_dataset import MolecularDataset
import logging
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split

logging.basicConfig(
    filename='evaluate_internal_split.log',
    filemode='w',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)
os.makedirs("node_outputs_raw", exist_ok=True)

# Config
batch_size = 32
epochs = 200
hidden_channels = 64
dropout_rate = 0.5
lr = 1e-3
node_block = "ABMP"
input_dir = "evaluate_model_outputs"
os.makedirs(input_dir, exist_ok=True)

dataset = pd.read_csv('../data/bace.csv')
smiles = dataset['SMILES'].tolist()
labels = dataset['Actividad'].tolist()
names = dataset['Title'].tolist()
smiles_train, smiles_blind, labels_train, labels_blind, names_train, names_blind = train_test_split(
    smiles, labels, names, test_size=0.2, random_state=42, stratify=labels
)

# === Dataset Preparation ===
train_dataset = MolecularDataset(smiles_train, names_train, labels_train)
eval_dataset = MolecularDataset(smiles_blind, names_blind, labels_blind)

global_dim = train_dataset.global_dim
edge_dim = train_dataset.edge_dim
num_node_features = train_dataset.num_node_features

successful_labels_train = train_dataset.successful_labels
successful_smiles_train = train_dataset.successful_smiles
successful_names_train = train_dataset.successful_names

if len(successful_labels_train) == 0:
    raise ValueError("No successful labels.")

weights = 1. / np.bincount(successful_labels_train)
samples_weights = weights[successful_labels_train]
sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)

train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
eval_loader = GeometricDataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# === Trainer Setup ===
trainer = GNNTrainer(
    smiles_list=successful_smiles_train,
    labels=successful_labels_train,
    names_list=successful_names_train,
    hidden_channels=hidden_channels,
    num_node_features=num_node_features,
    global_dim=global_dim,
    lr=lr,
    edge_dim=edge_dim,
    batch_size=batch_size,
    node_block=node_block,
    dropout_rate=dropout_rate,
    output_dir=input_dir
)
trainer.setup_model()

# === Train Model ===
for epoch in range(epochs):
    loss = trainer.train(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# === Evaluate ===
_, _, _, _, _, all_targets, all_preds, compound_names, all_preds_binarized = trainer.evaluate(eval_loader, generate_images=True)

# === Report ===
for name, true_label, pred_label in zip(compound_names, all_targets, all_preds_binarized):
    logger.info(f'Compound: {name}, True Label: {true_label}, Predicted Label: {pred_label}')
f1 = f1_score(all_targets, all_preds_binarized)
accuracy = accuracy_score(all_targets, all_preds_binarized)
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# === Confusion Matrix ===
cm = confusion_matrix(all_targets, all_preds_binarized)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix for Blind Test Set")
plt.savefig(os.path.join(input_dir, "conf_matrix.png"))

# === ROC Curve ===
fpr, tpr, _ = roc_curve(all_targets, all_preds)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Blind Test Set')
plt.legend(loc="lower right")
plt.savefig(os.path.join(input_dir, "roc_auc_blind_set.png"))


