# BMPNNs

**BMPNNs** is a modular Graph Neural Network (GNN) framework designed for molecular property prediction built on the Metalayer framework (Battaglia et. al., 2018). 
It supports multiple node-block models having in common a bidirectional message-passing. Options include attention mechanism applied to the message (similar to GAT), convolutional normalization, skip or include raw nodes connections.


| Node-Models | Description                                    |
| ----------- | ---------------------------------------------- |
| `BMP`       | Base message passing without self-nodes        |
| `ABMP`      | Attention-enhanced message passing             |
| `CBMP`      | Convolutional-normalization message passing    |
| `BMP+SN`    | BMP with self-nodes                            |
| `ABMP+SN`   | ABMP with self-nodes                           |

--------------------------------------------------------------------------------------
**INSTALLATION**

PyTorch Geometric (PyG) and its extensions require specific installation steps due to version-specific CUDA bindings. Follow the tested working environment setup below for best results using python 3.11 with GPU implementation using CUDA 12.8 and currently supported torch version (2.8).
 *These versions reflect the latest tested configuration. In future releases, PyG may support newer versions of CUDA and PyTorch. Please refer to the official PyG installation guide to adapt accordingly.

1. Create and activate a clean conda environment:
```bash

   conda create -n torch_gpu python=3.11 -y

   conda activate torch_gpu
```

2. Install PyTorch with CUDA 12.8:
```bash

   pip3 install torch torchvision
```

4. Install PyG extensions (must match your CUDA + PyTorch version):
```bash

   pip install torch_geometric
   
   # Optional dependencies:
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

5. Finally, install the package locally:
```bash

   pip install git+https://github.com/chemdesign-accl/BMPNNs.git
```
   or for development: 
```bash
   
git clone git@github.com:chemdesign-accl//BMPNNs.git
cd BMPNNs/
pip install -e .
```
--------------------------------------------------------------------------------------
