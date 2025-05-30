## Install `uv` (Python package manager)
https://docs.astral.sh/uv/getting-started/installation/

If on Mac, install using Homebrew:
```sh
brew install uv
```

## Install dependencies
```sh
# Instead of pip install -r requirements.txt
uv sync
```

## Add a new dependency
```sh
# Instead of pip install <name>
uv add <name>
```

## Remove a dependency
```sh
# Instead of pip uninstall <name>
uv remove <name>
```

## Reproduce the results
1. Run `download.ipynb` to download the dataset and save to CSV format.
2. Run `preproc.ipynb` to clean and preprocesses the CSV file.
3. `run_test.py` calculates p-values for each peptide's relative intensity compared to background variation distribution and performs statistical testing to identify significantly affected variants.
4. `run_protein_drug_interaction.py` compares the per-drug background variation distribution for the most frequently observed protein (Q8TD19) and generates distribution plots for each drug concentration.
5. `modification_level.ipynb` creates a heatmap of top peptide modifications to drug treaments, ranked by total absolute response.
6. `protein_drug_graph.ipynb` performs protein-drug interaction analysis on the number of drugs affecting proteins.
7. `monotonic.py` identifies dose-response relationships by finding variants that show strictly monotonic intensity changes (increasing or decreasing) across drug concentrations (3nM → 300nM → 3000nM → 30000nM).
8. `monotonic_proteins.py` checks for monotonic dose-response patterns across average intensities of proteins.
