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
2. Run `preproc.ipynb` to clean the CSV file.
3. `run_test.py` calculates all p-values of each peptide's relative intensity against the background variation distribution
4. `run_protein_level.py` runs protein level analysis to see differences in the backgroun variation distribution and specific treated protein distribution
