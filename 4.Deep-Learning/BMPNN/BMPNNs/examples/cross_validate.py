import pandas as pd
import logging
import os
from BMPNNs import GNNTrainer
logging.basicConfig(
    filename='cross_validate.log',
    filemode='w',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
input_dir = "cross_validate_output"
os.makedirs(input_dir, exist_ok=True)
data = pd.read_csv('../data/bbbp.csv')
smiles_train = data['SMILES'].tolist()
labels_train = data['Actividad'].tolist()
names_train = data['Title'].tolist()
trainer = GNNTrainer(
    smiles_list=smiles_train,
    labels=labels_train,
    names_list=names_train,
    hidden_channels=20,
    lr=0.003,
    batch_size=17,
    k_folds=5,
    dropout_rate=0.08,
    input_dir = input_dir
)
trainer.cross_validate()
