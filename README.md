YAMMBS
======

Yet Another Molecular Mechanics Benchmaring Suite (YAMMBS, pronounced like "yams") is a tool for
benchmarking force fields.

YAMMBS is currently developed for internal use at Open Force Field. It is not currently recommended for external use. No guarantees are made about the stability of the API or the accuracy of any results.

# Getting started

## Installation

Use the file `./devtools/conda-envs/dev.yaml` and also install `yammbs` with something like `python -m pip install -e .`.

## Data sources

It is assumed that the input molecules are stored in a `openff-qcsubmit` model like `OptimizationResultCollection`.

## Key API points

See the file `run.py` for a start-to-finish example.

Load a molecule dataset into the used representation:

```python
from yammbs import MoleculeStore

store = MoleculeStore.from_qcsubmit_collection(
    collection=my_collection,
    database_name="my_database.sqlite",
)
```

Run MM optimizations of all molecules using a particular force field

```python
store.optimize_mm(force_field="openff-2.1.0.offxml")
```

Run DDE (or RMSD, TFD, etc.) analyses and save to results disk:

```python
ddes = store.get_dde(force_field="openff-2.1.0.offxml")
ddes.to_csv(f"{force_field}-dde.csv")
```

Note that the pattern in the script

```python
from multiprocessing import freeze_support

def main():
    # Your code here

if __name__ == "__main__":
    freeze_support()
    main()
```

must be used for Python's `multiprocessing` module to behave well.

### License

YAMMBS is open-source software distrubuted under the MIT license (see LICENSE). It derives from
other open-source work that may be distributed under other licenses (see LICENSE-3RD-PARTY).

### Copyright

Copyright (c) 2022, Open Force Field Initiative
