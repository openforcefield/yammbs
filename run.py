import pathlib
from multiprocessing import freeze_support

import numpy
import pandas
from matplotlib import pyplot
from openff.qcsubmit.results import OptimizationResultCollection

from yammbs import MoleculeStore
from yammbs.inputs import QCArchiveDataset


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

    data = "ch"

    dataset = QCArchiveDataset.from_qcsubmit_collection(
        OptimizationResultCollection.parse_file(
            "yammbs/_tests/data/qcsubmit/01-processed-qm-ch.json",
        ),
    )

    if pathlib.Path(f"{data}.sqlite").exists():
        store = MoleculeStore(f"{data}.sqlite")
    else:
        store = MoleculeStore.from_qm_dataset(
            dataset,
            database_name=f"{data}.sqlite",
        )

    for force_field in force_fields:
        # This is called within each analysis method, but short-circuiting within them. It's convenient to call it here
        # with the freeze_support setup so that later analysis methods can trust that the MM conformers are there
        store.optimize_mm(force_field=force_field)

        store.get_dde(force_field=force_field).to_csv(f"{force_field}-dde.csv")
        store.get_rmsd(force_field=force_field).to_csv(f"{force_field}-rmsd.csv")
        store.get_tfd(force_field=force_field).to_csv(f"{force_field}-tfd.csv")

    plot(force_fields)


def plot(force_fields: list[str]):
    x_ranges = {
        "dde": (-16.0, 16.0),
        "rmsd": (-0.3, 3.3),
        "tfd": (-0.05, 0.55),
    }
    for data in ["dde", "rmsd", "tfd"]:
        figure, axis = pyplot.subplots()
        for force_field in force_fields:
            dataframe = pandas.read_csv(f"{force_field}-{data}.csv")

            if data == "dde":
                counts, bins = numpy.histogram(
                    dataframe[dataframe.columns[-1]],
                    bins=numpy.linspace(-15, 15, 31),
                )

                axis.stairs(counts, bins, label=force_field)

                axis.set_ylabel("Count")

            else:
                sorted_data = numpy.sort(dataframe[dataframe.columns[-1]])

                axis.plot(
                    sorted_data,
                    numpy.arange(1, len(sorted_data) + 1) / len(sorted_data),
                    "-",
                    label=force_field,
                )

                axis.set_xlabel(data)
                axis.set_ylabel("CDF")

                axis.set_xlim(x_ranges[data])
                axis.set_ylim((-0.05, 1.05))

        axis.legend(loc=0)

        figure.savefig(f"{data}.png", dpi=300)


if __name__ == "__main__":
    # This setup is necessary for reasons that confused me - both setting it up in the __main__ block and calling
    # freeze_support(). This is probably not necessary after MoleculeStore.optimize_mm() is called, so you can load up
    # the same database for later analysis once the MM conformers are stored
    freeze_support()
    main()
