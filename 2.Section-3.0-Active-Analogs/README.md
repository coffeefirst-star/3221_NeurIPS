# Active analog and class-separation analysis

This folder contains the analysis used to inspect whether the selected inactive/decoy compounds are trivially separable from the active compounds. The goal is to quantify active-decoy chemical similarity and to generate benchmark datasets with controlled fractions of decoys that fall in active-containing BitBIRCH clusters.

## Files

- `actives_prop_decoys.py`: command-line script for active/decoy BitBIRCH clustering, cluster-composition analysis, and benchmark generation.
- `actives_left_outs_analysis.ipynb`: notebook used to inspect active analogs and active compounds left out by dataset construction choices.

## Required inputs

The main script expects two CSV files:

- active compounds CSV
- decoy/inactive compounds CSV

Each CSV must contain molecular structures and identifiers. By default the script searches for common column names:

| Required value | Accepted/default column names |
|---|---|
| SMILES | `SMILES`, `Smiles`, `smiles` |
| Title/ID | `Title`, `title`, `Sample_ID`, `ID`, `id`, `Name` |
| Label | optional, because labels are assigned from the input file role |

Actives are assigned `Label = 1`; decoys are assigned `Label = 0`.

## Method summary

1. Canonicalize SMILES with RDKit and remove invalid or duplicate canonical SMILES.
2. Compute 2048-bit ECFP4 fingerprints using `bblean`.
3. Cluster combined active and decoy fingerprints with BitBIRCH.
4. Assign each molecule to a cluster and compute cluster composition:
   - pure active clusters
   - mixed active/decoy clusters
   - pure decoy clusters
5. Build benchmark datasets with different fractions of decoys sampled from mixed active-containing clusters.
6. Plot active-decoy cluster composition and the Tanimoto similarity distribution for decoys in mixed clusters.

This analysis supports the dataset motivation by documenting how many decoys are analog-like or chemically close to active compounds, rather than relying only on easily separable random decoys.

## Example command

```bash
python actives_prop_decoys.py \
  --actives-csv actives_tagged.csv \
  --decoys-csv inactives_tagged.csv \
  --ratio-k 1.0 \
  --random-state 42 \
  --clusters-out bitbirch_active_decoy_clusters.csv \
  --venn-plot active_cluster_composition.png
```

Use `--actives-smiles-col`, `--decoys-smiles-col`, `--actives-title-col`, and `--decoys-title-col` if the input CSVs use non-standard column names.

## Outputs

The script writes:

- `bitbirch_active_decoy_clusters.csv` or the path passed to `--clusters-out`: all active and decoy compounds with assigned cluster IDs and cluster composition counts.
- `active_cluster_composition.png` or the path passed to `--venn-plot`: cluster-composition summary plot.
- `mixed_decoys_tanimoto_histogram.png`: distribution of maximum Tanimoto similarity from mixed-cluster decoys to actives.
- `benchmark_bitbirch_0pct_mixed.csv`
- `benchmark_bitbirch_25pct_mixed.csv`
- `benchmark_bitbirch_50pct_mixed.csv`
- `benchmark_bitbirch_75pct_mixed.csv`
- `benchmark_bitbirch_100pct_mixed.csv`
- corresponding `benchmark_bitbirch_*pct_mixed_only.csv` files containing actives plus only the selected mixed-cluster decoys.

The `0pct` to `100pct` files vary the fraction of selected decoys drawn from mixed active-containing clusters. These files can be used to study how model performance changes as decoys become more chemically similar to actives.

## Dependencies

This analysis uses:

- `numpy`
- `pandas`
- `rdkit`
- `bblean` for BitBIRCH clustering: https://github.com/mqcomplab/bblean
- `iSIM`for similarity: https://github.com/mqcomplab/iSIM
- `matplotlib`
- `matplotlib-venn`

## Notes

- Fingerprints are ECFP4, equivalent to Morgan radius 2, with 2048 bits.
- BitBIRCH clustering uses packed fingerprints internally.
- The script estimates a similarity threshold from representative samples and then reclusters for refinement.
- Large decoy CSVs may require substantial memory.
