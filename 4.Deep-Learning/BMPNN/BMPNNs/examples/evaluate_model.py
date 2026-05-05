
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader as GeometricDataLoader
import BMPNNs
import inspect
print(inspect.getfile(BMPNNs))
import time
from BMPNNs import GNNTrainer
from BMPNNs.data.molecular_dataset import MolecularDataset
import logging
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

logging.basicConfig(
    filename='evaluation.log',
    filemode='w',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)
os.makedirs("node_outputs_raw", exist_ok=True)

threshold = 0
batch_size = 32
epochs = 50
hidden_channels = 250
dropout_rate = 0.25
lr= 0.003
node_block = "ABMP"
input_dir = "evaluate_model_outputs"
os.makedirs(input_dir, exist_ok=True)

# === Load Data ===
train_df = pd.read_csv('../BMPNNs/data/bace.csv')
eval_df = pd.read_csv('../BMPNNs/data/TRPA1_for_evaluation.csv')

smiles_train = train_df['SMILES'].tolist()
labels_train = train_df['Actividad'].tolist()
names_train = train_df['Title'].tolist()

smiles_eval = eval_df['SMILES'].tolist()
labels_eval = eval_df['Actividad'].tolist()
names_eval = eval_df['Title'].tolist()

# === Dataset Preparation ===
train_dataset = MolecularDataset(smiles_train, names_train, labels_train, node_block)
#eval_dataset = MolecularDataset(smiles_eval, names_eval, labels_eval)

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
#eval_loader = GeometricDataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

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
    input_dir=input_dir
)
print(f"Using trainer: {type(trainer).__name__}")

trainer.setup_model()

# === Train Model ===
start_time = time.time()  # Start full training timer

for epoch in range(epochs):
    epoch_start = time.time()  # Start epoch timer
    loss = trainer.train(train_loader)
    epoch_time = time.time() - epoch_start  # Time for this epoch

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Time: {epoch_time:.2f}s")

total_time = time.time() - start_time  # Total training time
print(f"\nTotal training time: {total_time:.2f}s")
"""

# === Evaluate ===
_, _, _, _, _, all_targets, all_preds, compound_names, all_preds_binarized = trainer.evaluate(eval_loader, generate_images=True)

# === Report ===
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
"""