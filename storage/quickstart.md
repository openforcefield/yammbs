# Getting started

## Installation

Use the file `../devtools/conda-envs/dev.yaml` and also install `ibstore` with something like `python -m pip install -e .`.

## Data source

It is assumed that the input molecules are stored in a `openff-qcsubmit` model like `OptimizationResultCollection`.

## Key API points

See the file `run.py` for a start-to-finish example.

Load a molecule dataset into the used representation:

```python
from ibstore import MoleculeStore

store = MoleculeStore.from_qcsubmit_collection(
    collection=my_collection,
    database_name="my_database.sqlite",
)
```

Run MM optimizations of all molecules using a particular force field

```python
store.optimize_mm(force_field="openff-2.1.0.offxml")
```

Run DDE or RMSD analysis and save to disk:

```python
ddes = store.get_dde(force_field="openff-2.1.0.offxml")
ddes.to_csv(f"{force_field}-dde.csv")

rmsds = store.get_rmsd(force_field="openff-2.1.0.offxml")
rmsds.to_csv(f"{force_field}-rmsd.csv")
```

Note that the pattern

```python
def main():
    # Your code here

if __name__ == "__main__":
    freeze_support()
    main()
```

must be used for Python's `multiprocessing` module to behave well.
