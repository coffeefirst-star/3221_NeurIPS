# Initial VS library clustering

This folder contains the scripts used to reduce large virtual-screening libraries into representative medoids before dataset construction. The initial clustering step converts raw molecules into 2048-bit ECFP4 fingerprints and clusters them with BitBIRCH. The resulting medoids define the purchasable virtual-screening space used by the downstream sampling approaches.

The VS libraries were used to build a representative purchasable space and are characterized as mostly drug-like, diverse when available from the vendor, and available in stock or on demand when indicated.

## Scripts

Using SDF files:

- `1-sdf-to-npy.py`: converts SDF files into packed 2048-bit ECFP4 fingerprint chunks saved as `fingerprints.npy`.
- `2-clustering-sdf.py`: reads existing `*.npy` packed fingerprint chunks, clusters them with BitBIRCH, writes cluster IDs to `clustered_ids_parallel.pkl`, and writes `npy_medoids.pkl`.

Directly from smiles:

- `2-clustering.py`: reads one or more `*.smi` files, computes packed 2048-bit ECFP4 fingerprints with `bblean`, clusters them with BitBIRCH, and writes `npy_medoids.pkl`.

UF-Scripps actives and decoys that are already stored as tagged CSV files can be transformed directly to a BitBIRCH-compatible `.pkl` format using:

- `make_pkl.py`: reads a CSV with `Title` and `SMILES`, computes unpacked 2048-bit ECFP4 fingerprints, and stores each compound as a medoid with `cluster_size = 1`.

The output expected by downstream code is:

```text
Libraries/<library_name>/npy_medoids.pkl
```

The medoid pickle stores:

- `fingerprints`: unpacked medoid fingerprints.
- `medoid_indices`: selected medoid indices in the source fingerprint array.
- `cluster_size`: number of original fingerprints represented by each medoid.

## Required BitBIRCH dependency

BitBIRCH is provided by the `bblean` package:

```text
https://github.com/mqcomplab/bblean
```

Install it with:

```bash
pip install bblean
# Alternatively, with uv:
uv pip install bblean
bb --help
```

## Fingerprints and clustering settings

- Fingerprint type: ECFP4, equivalent to Morgan radius 2.
- Fingerprint length: 2048 bits.
- Packed fingerprints are used internally for memory efficiency.
- BitBIRCH `branching_factor`: 50.
- BitBIRCH `merge_criterion`: `diameter`.
- Initial threshold: `mean(pairwise representative similarity) + 3.5 * std(pairwise representative similarity)`.
- Refinement: 5 reclustering iterations with `extra_threshold = std`.



## Table A1. Initial library clustering summary

| VS Library | Fingerprints | Medoids | Tag | Source |
|---|---:|---:|---|---|
| WuXi | 166,969,544 | 406,617 | in-stock/on-demand | Zinc15 |
| Akos | 23,781,377 | 383,811 | in-stock/on-demand | Zinc15 |
| Princeton | 1,532,308 | 76,136 | Diverse collection | Princeton |
| ChemDiv | 300,000 | 35,010 | Diverse | ChemDiv |
| Enamine | 4,573,361 | 428,101 | Diverse | Enamine |
| ChemBL | 2,854,801 | 275,441 | Bioactive and Drug-like | EBI |
| MolPort | 2,474,577 | 310,752 | Diverse | MolPort |
| Life Chemicals | 195,840 | 18,221 | Diverse | Life Chemicals |
| MCule | 43,896,819 | 615,154 | in-stock/on-demand | Zinc15 DB |
| Asinex | 408,357 | 26,553 | Diverse | Asinex |
| Otava | 9,975 | 3,037 | Diverse | Otava |
| Coconut | 695,115 | 36,354 | Natural compounds | Coconut |
| Scubidoo | 999,794 | 39,108 | Mixed building blocks | Scubidoo |
| Ambeed | 674,598 | 75,356 | in-stock/on-demand | Zinc15 |
| Bldpharm | 664,818 | 66,553 | in-stock/on-demand | Zinc15 |
| Innova | 1,566,996 | 34,716 | In stock/on demand | Zinc15 |
| TOTAL | 251,598,280 | 2,830,920 |  |  |

## Notes for GitHub and dataset submission

The raw vendor libraries are very large and may have redistribution restrictions. For a public GitHub repository, include the scripts, metadata, and small derived CSVs needed for verification. Store large raw inputs and medoid PKLs in an archival data release when redistribution is permitted, and document external download sources otherwise.
