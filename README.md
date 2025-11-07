# YAMMBS

Yet Another Molecular Mechanics Benchmarking Suite (YAMMBS, pronounced like "yams") is a tool for
benchmarking force fields.

YAMMBS is currently developed for internal use at Open Force Field. It is not currently recommended for external use. No guarantees are made about the stability of the API or the accuracy of any results. Feedback and contributions are welcome [on GitHub](https://github.com/openforcefield/yammbs).

## Installation

Use the file `./devtools/conda-envs/dev.yaml` and also install `yammbs` with something like `python -m pip install -e .`.

## Getting started

See the file [run.py](run.py) for a start-to-finish example. Note that the pattern in the script

```python
from multiprocessing import freeze_support

def main():
    # Your code here

if __name__ == "__main__":
    freeze_support()
    main()
```

must be used for Python's `multiprocessing` module to behave well.

### Data sources

It is assumed that the input molecules are stored in an `openff-qcsubmit` model like `OptimizationResultCollection` or YAMMBS's own input models.

### Preparing an input dataset

YAMMBS relies on QCSubmit to provide datasets from QCArchive. See [their docs](https://docs.openforcefield.org/projects/qcsubmit/en/stable/), particularly the [dataset retrieval example](https://docs.openforcefield.org/projects/qcsubmit/en/stable/examples/retrieving-results.html), for more.

Currently only optimization datasets (`OptimizationResultCollection` in QCSubmit) are supported.

First, retrieve a dataset from QCArchive:

```python
from qcportal import PortalClient

from openff.qcsubmit.results import OptimizationResultCollection


client = PortalClient("https://api.qcarchive.molssi.org:443", cache_dir=".")

season1_dataset = OptimizationResultCollection.from_server(
    client=client,
    datasets="OpenFF Industry Benchmark Season 1 v1.1",
    spec_name="default",
)
```

After retrieving it - and after applying filters to remove problematic records - you can dump it to disk to avoid pulling down all of the data from the server again.

```python
with open("qcsubmit.json", "w") as f:
    f.write(season1_dataset.json())
```

Once an `OptimizationResultCollection` is in memory, either by pulling it down from QCArchive or loading it from disk, convert it to a "YAMMBS input" model using the API:

```python
from yammbs.inputs import QCArchiveDataset


season1_dataset = OptimizationResultCollection.parse_file("qcsubmit.json")

dataset = QCArchiveDataset.from_qcsubmit_collection(season1_dataset)

with open("input.json", "w") as f:
    f.write(dataset.model_dump_json())
```

This input model (`QCArchiveDataset`) stores a minimum amount of information to use these QM geometries as reference structures. The dataset has fields for tagging the name and model version, but mostly stores a list of structures. Each QM-optimized structure is stored as a `QCArchiveMolecule` object which stores:

* (Mapped) SMILES which can be used to regenerate the `openff.toolkit.Molecule` and similar objects
* QM-optimized geometry
* Final energy from QM optimization
* An ID uniquely defining this structure within the datasets

If running many benchmarks, we recommend using this file as a starting point.

Note: This JSON file ("input.json") is from a different model than the JSON file written from QCSubmit - they are not interchangeable.

Note: Both QCSubmit and YAMMBS rely on Pydantic for model validation and serialization. Even though both use V2 in packaging, YAMMBS uses the V2 API and (as of October 2024) QCSubmit still uses the V1 API. Usage like above should work fine; only esoteric use cases (in particular, defining a new model that has both YAMMBS and QCSubmit models as fields) should be unsupported.

### Run a benchmark

With the input prepared, create a `MoleculeStore` object:

```python
from yammbs import MoleculeStore

store = MoleculeStore.from_qcarchive_dataset(dataset)
```

This object is the focal point of running benchmarks; it stores the inputs (QM structures), runs minimizations with force field(s) of interest, stores the results (MM structures), and provides helper methods for use in analysis.

Run MM optimizations of all molecules using a particular force field(s) using `optimize_mm`:

```python
store.optimize_mm(force_field="openff-2.1.0.offxml")

# can also iterate over multiple force fields, and use more processors
for force_field in [
    "openff-1.0.0.offxml",
    "openff-1.3.0.offxml",
    "openff-2.0.0.offxml",
    "openff-2.1.0.offxml",
    "openff-2.2.1.offxml",
    "gaff-2.11",
    "de-force-1.0.1.offxml",
]:
    store.optimize_mm(force_field=force_field, n_processes=16)
```

This method short-circuits (i.e. does not run minimizations) if a force field's results are already stored. i.e. the Sage 2.1 optimizations in the loop will be skipped.

There are "output" models that mirror the input models, basically storing MM-minimized geometries without needing to re-load or re-optimize the QM geometries. This can again be saved out to disk as JSON:

```python
store.get_outputs().model_dump_json("output.json")
```

Summary metrics (including DDE, RMSD, TFD, and internal coordinate RMSDs) are available separately (in order to reduce file size when only summary statistics, and not whole molecular descriptions and geometries, are sought):

```python
store.get_metrics().model_dump_json("metrics.json")
```

The basic structure of the metrics is a hierarchical dictionary. It is keyed by force field tag (i.e. "openff-2.2.1") mapping on to a dict of per-molecule summary metrics. Each of these dicts are keyed by QCArchvie ID (the same ID used to distinguish structures in the input and output models) mapping onto a dict of string-float keys that store the actual metrics (i.e. the DDE, RMSD, etc. of this particular structure optimized with the force field used as its high-level key). Access to these data is similar in memory (on the Pydantic models) and on disk (in JSON). Visually:

```json
{
    "metrics": {
        "openff-1.0.0": {
            "37016854": {
                "dde": 0.5890449032115157,
                "rmsd": 0.011969891530473157,
                "tfd": 0.001592046369769131,
                "icrmsd": {
                    "Bond": 0.0033974261816308144,
                    "Angle": 0.9483605366613115,
                    "Dihedral": 1.353163675708829,
                    "Improper": 0.2922040744956022,
                },
            },
            "37016855": {"this molecule's metrics ..."},
        },
        "openff-2.0.0": {
            "37016855": {"this force field's molecules ..."},
        }
    }
}
```

This data can be transformed for plotting, summary statistics, etc. which compare the metrics of each force field (for this molecule dataset).

### Run a TorsionDrive benchmark

`YAMMBS` contains functionality for running TorsionDrive benchmarks in `yammbs.torsion`. Also, a convenience script for torsion analysis is provided in `yammbs/scripts/run_torsion_comparisons.py`. This can be run from anywhere using `yammbs_analyse_torsions`:

```bash
yammbs_analyse_torsions --help
```

For example, to run the [OpenFF Rowley Biaryl v1.0 TorsionDrive dataset](https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2020-06-17-OpenFF-Biaryl-set), first download it with

```python
from openff.qcsubmit.results import TorsionDriveResultCollection
from qcportal import PortalClient

from yammbs.torsion.inputs import QCArchiveTorsionDataset

client = PortalClient("https://api.qcarchive.molssi.org:443", cache_dir=".")

rowley_torsion_dataset = TorsionDriveResultCollection.from_server(
    client=client,
    datasets="OpenFF Rowley Biaryl v1.0",
    spec_name="default",
)

dataset = QCArchiveTorsionDataset.from_qcsubmit_collection(rowley_torsion_dataset)

with open("input.json", "w") as f:
    f.write(dataset.model_dump_json())
```

Then, to run the benchmark with `openff-1.0.0.offxml` and `openff-2.2.1.offxml`:

```bash
yammbs_analyse_torsions --qcarchive-torsion-data input.json \
    --base-force-fields openff-1.0.0 \
    --base-force-fields openff-2.2.1
```
This takes a bit over 10 minutes on a 32-core machine with 125 GB RAM. Note that when supplying your own force fields, make sure that these are the unconstrained versions (this is done automatically for e.g. `openff-1.0.0`), for example:

```bash
yammbs_analyse_torsions --qcarchive-torsion-data input.json \
    --extra-force-fields my_unconstrained_ff.offxml
```
A range of OpenFF force fields will be run for comparison if no `--base-force-fields` are specified.

## Custom analyses

See [examples.ipynb](examples.ipynb) for some examples of interacting with benchmarking results and a starting point for custom analyses.

### License

YAMMBS is open-source software distrubuted under the MIT license (see LICENSE). It derives from
other open-source work that may be distributed under other licenses (see LICENSE-3RD-PARTY).

### Copyright

Copyright (c) 2022, Open Force Field Initiative
