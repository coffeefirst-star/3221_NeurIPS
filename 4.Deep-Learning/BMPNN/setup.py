from setuptools import setup, find_packages

setup(
    name='bmpnns',
    version='0.1',
    description='MPNNs for molecular prediction',
    author='Alma C. Castaneda-Leautaud',
    packages=find_packages(),
    install_requires=[
        "python-dateutil",
        "packaging",
        "rdkit",
        "pandas",
        "matplotlib",
        "numpy",
        "tqdm",
        "Pillow",
        "molvs",
        "mendeleev",
        "scikit-learn",
        "torch>=2.7.0",
        "torchvision>=0.21.0",
        "torchaudio>=2.7.0",
        "pyg-lib",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        
    ],
    python_requires='>=3.10'
)

