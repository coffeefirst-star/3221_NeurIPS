import os
import pandas as pd
import numpy as np
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader as GeometricDataLoader
from BMPNNs import GNNTrainer
from BMPNNs.data.molecular_dataset import MolecularDataset
import logging
logging.basicConfig(
    filename='predict.log',
    filemode='w',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)
import time
start = time.time()
train_df = pd.read_csv('../data/TRPA1_for_training.csv')
df = pd.read_csv('../data/bace.csv', dtype={0: str}) #dummy test set

batch_size = 155
epochs = 150
hidden_channels = 217
dropout_rate = 0.26
lr = 1e-3
node_block = "ABMP+SN" #Options: BMP, ABMP, CBMP or BMP+SN
input_dir = "evaluate_model_outputs"
os.makedirs(input_dir, exist_ok=True)

smiles_train = train_df['SMILES'].tolist()
labels_train = train_df['Actividad'].tolist()
names_train = train_df['Title'].tolist()

smiles_eval = df['SMILES'].tolist()
labels_eval = df['Actividad'].tolist()
names_eval = df['Title'].tolist()

# === Dataset Preparation ===
train_dataset = MolecularDataset(smiles_train, names_train, labels_train)
eval_dataset = MolecularDataset(smiles_eval, names_eval, labels_eval)

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
    lr=lr,
    batch_size=batch_size,
    dropout_rate=dropout_rate,
    input_dir=input_dir,
    node_block="ABMP"
)
trainer.setup_model()

# === Train Model ===
for epoch in range(epochs):
    loss = trainer.train(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    

smiles_list = df['SMILES'].tolist()
compounds_list = df['Title'].tolist()
output_csv_path = 'predictions.csv'
results = trainer.predict(smiles_list, compounds_list, output_csv_path)
end = time.time()
elapsed = end - start
print(f"Elapsed time: {elapsed:.2f} seconds")