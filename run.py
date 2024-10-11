import pathlib
from multiprocessing import freeze_support

import numpy
from matplotlib import pyplot

from yammbs import MoleculeStore
from yammbs.inputs import QCArchiveDataset
from yammbs.outputs import MetricCollection


def main():
    force_fields = [
        "openff-1.0.0",
        "openff-1.1.0",
        "openff-1.2.0",
        "openff-1.3.0",
        "openff-2.0.0",
        "openff-2.1.0",
        "gaff-1.8",
        "gaff-2.11",
        "de-force-1.0.1.offxml",
    ]

    # pre-processed data is stored in the repository, just a larger dataset
    # filtered by elements, takes values of "ch" (~40) and "cho" (~1600 molecules)
    data_key = "ch"

    dataset = QCArchiveDataset.model_validate_json(f"yammbs/_tests/data/yammbs/01-processed-qm-{data_key}.json")

    # This file is also store at f"yammbs/_tests/data/yammbs/01-processed-qm-{data_key}.json"
    if pathlib.Path(f"{data_key}.sqlite").exists():
        store = MoleculeStore(f"{data_key}.sqlite")
    else:
        store = MoleculeStore.from_qm_dataset(
            dataset,
            database_name=f"{data_key}.sqlite",
        )

    for force_field in force_fields:
        # This is called within each analysis method, but short-circuiting within them. It's convenient to call it here
        # with the freeze_support setup so that later analysis methods can trust that the MM conformers are there
        store.optimize_mm(force_field=force_field, n_processes=10)

    with open("minimized.json", "w") as f:
        f.write(store.get_outputs().model_dump_json())

    with open("metrics.json", "w") as f:
        f.write(store.get_metrics().model_dump_json())

    plot(force_fields)


def plot(force_fields: list[str]):
    metrics = MetricCollection.parse_file("metrics.json")

    x_ranges = {
        "dde": (-16.0, 16.0),
        "rmsd": (-0.3, 3.3),
        "tfd": (-0.05, 0.55),
    }

    # metrics are stored with force field at the top of the hierarchy,
    # restructure it so that the type of metric is at the top

    # these each keep the qcarchive_id in case it's useful, though they're not
    # used in this script
    ddes: dict[str, dict[int, float]] = {
        force_field: {key: val.dde for key, val in metrics.metrics[force_field].items()}
        for force_field in metrics.metrics.keys()
    }

    rmsds = {
        force_field: {key: val.rmsd for key, val in metrics.metrics[force_field].items()}
        for force_field in metrics.metrics.keys()
    }

    tfds = {
        force_field: {key: val.tfd for key, val in metrics.metrics[force_field].items()}
        for force_field in metrics.metrics.keys()
    }

    data = {
        "dde": ddes,
        "rmsd": rmsds,
        "tfd": tfds,
    }
    for key in ["dde", "rmsd", "tfd"]:
        figure, axis = pyplot.subplots()

        for force_field in force_fields:
            if key == "dde":
                counts, bins = numpy.histogram(
                    [*data[key][force_field].values()],
                    bins=numpy.linspace(-15, 15, 31),
                )

                axis.stairs(counts, bins, label=force_field)

                axis.set_ylabel("Count")

            else:
                sorted_data = numpy.sort([*data[key][force_field].values()])

                axis.plot(
                    sorted_data,
                    numpy.arange(1, len(sorted_data) + 1) / len(sorted_data),
                    "-",
                    label=force_field,
                )

                axis.set_xlabel(key)
                axis.set_ylabel("CDF")

                axis.set_xlim(x_ranges[key])
                axis.set_ylim((-0.05, 1.05))

        axis.legend(loc=0)

        figure.savefig(f"{key}.png", dpi=300)


if __name__ == "__main__":
    # This setup is necessary for reasons that confused me - both setting it up in the __main__ block and calling
    # freeze_support(). This is probably not necessary after MoleculeStore.optimize_mm() is called, so you can load up
    # the same database for later analysis once the MM conformers are stored
    freeze_support()
    main()
