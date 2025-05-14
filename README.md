## Install Python package manager
https://docs.astral.sh/uv/getting-started/installation/

If on Mac, install using Homebrew:
```sh
brew install uv
```

## Install dependencies
```sh
uv sync
```

## Add a new dependency
```sh
# Instead of pip install <name>
uv add <name>
```

## Reproduce the results
1. Run `download.ipynb` to download the dataset and save to CSV format.
2. Run `preproc.ipynb` to clean the CSV file.
3. `run_test.ipynb` conducts a T-test for a specific protein and drug.
