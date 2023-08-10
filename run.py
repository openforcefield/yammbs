from multiprocessing import freeze_support

import numpy
import pandas
from matplotlib import pyplot
from openff.qcsubmit.results import OptimizationResultCollection

from ibstore._store import MoleculeStore


def main():
    force_fields = [
        f"openff-{val}.0"
        for val in [
            "2.1",
        ]
    ]

    store = MoleculeStore.from_qcsubmit_collection(
        OptimizationResultCollection.parse_file(
            "ibstore/_tests/data/01-processed-qm-ch.json",
        ),
        # TODO: Don't assume this file doesn't already exist
        database_name="tmp.sqlite",
    )

    for force_field in force_fields:
        # This is called within each analysis method, but short-circuiting within them. It's convenient to call it here
        # with the freeze_support setup so that later analysis methods can trust that the MM conformers are there
        store.optimize_mm(force_field=force_field)

        store.get_dde(force_field=force_field).to_csv(f"{force_field}-dde.csv")
        store.get_rmsd(force_field=force_field).to_csv(f"{force_field}-rmsd.csv")
        store.get_tfd(force_field=force_field).to_csv(f"{force_field}-tfd.csv")

    plot_cdfs(force_fields)


def plot_cdfs(force_fields):
    x_ranges = {
        "dde": (-5.0, 5.0),
        "rmsd": (0.0, 4.0),
        "tfd": (0.0, 2.0),
    }
    for data in ["dde", "rmsd", "tfd"]:
        figure, axis = pyplot.subplots()
        for force_field in force_fields:
            dataframe = pandas.read_csv(f"{force_field}-{data}.csv")

            sorted_data = numpy.sort(dataframe[dataframe.columns[-1]])

            axis.plot(
                sorted_data,
                numpy.arange(1, len(sorted_data) + 1) / len(sorted_data),
                ".--",
                label=force_field,
            )

        axis.set_xlabel(data)
        axis.set_ylabel("CDF")

        axis.set_xlim(x_ranges[data])
        axis.set_ylim((0.0, 1.0))

        axis.legend(loc=0)

        figure.savefig(f"{data}.png")


if __name__ == "__main__":
    # This setup is necessary for reasons that confused me - both setting it up in the __main__ block and calling
    # freeze_support(). This is probably not necessary after MoleculeStore.optimize_mm() is called, so you can load up
    # the same database for later analysis once the MM conformers are stored
    freeze_support()
    main()
