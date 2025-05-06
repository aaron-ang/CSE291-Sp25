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
