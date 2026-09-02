# Installation

## Allocation-only environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Training

```bash
python -m pip install -e ".[train]"
```

## COCO evaluation and plotting

```bash
python -m pip install -e ".[eval,plots]"
```

## Development

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/release_check.py
```
