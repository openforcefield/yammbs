import pathlib
from multiprocessing import freeze_support

import click
import numpy as np
from matplotlib import pyplot

from yammbs.torsion import TorsionStore
from yammbs.torsion.inputs import QCArchiveTorsionDataset
from yammbs.torsion.outputs import MetricCollection


@click.command()
@click.option(
    "--force-fields",
    "-f",
    multiple=True,
    default=["openff-1.0.0", "openff-2.0.0", "openff-2.1.0", "openff-2.2.0", "openff-2.2.1"],
    help="List of force fields to use for optimization.",
)
@click.option(
    "--qcarchive-torsion-data",
    "-i",
    default="qca-torsion-data.json",
    help="Input file containing torsion drive data in QCArchive json format.",
)
@click.option(
    "--database-file",
    "-d",
    default="torsion-data.sqlite",
    help="SQLite database file to store torsion data.",
)
@click.option(
    "--output-metrics",
    "-m",
    default="metrics.json",
    help="Output file for metrics.",
)
@click.option(
    "--output-minimized",
    "-o",
    default="minimized.json",
    help="Output file for minimized data.",
)
@click.option(
    "--plot-dir",
    "-p",
    default=".",
    help="Directory to save the generated plots.",
)
def main(
    force_fields: list[str],
    qcarchive_torsion_data: str,
    database_file: str,
    output_metrics: str,
    output_minimized: str,
    plot_dir: str,
) -> None:
    """
    Run torsion drive comparisons using specified force fields and input data.
    """
    dataset = QCArchiveTorsionDataset.model_validate_json(open(qcarchive_torsion_data).read())

    if pathlib.Path(database_file).exists():
        store = TorsionStore(database_file)
    else:
        store = TorsionStore.from_torsion_dataset(
            dataset,
            database_name=database_file,
        )

    for force_field in force_fields:
        store.optimize_mm(force_field=force_field, n_processes=24)

    if not pathlib.Path(output_minimized).exists():
        with open(output_minimized, "w") as f:
            f.write(store.get_outputs().model_dump_json())

    if not pathlib.Path(output_metrics).exists():
        with open(output_metrics, "w") as f:
            f.write(store.get_metrics().model_dump_json())

    fig, axes = pyplot.subplots(5, 4, figsize=(20, 20))

    for molecule_id, axis in zip(store.get_molecule_ids(), axes.flatten()):
        _qm = store.get_qm_energies_by_molecule_id(molecule_id)

        _qm = dict(sorted(_qm.items()))

        qm_minimum_index = min(_qm, key=_qm.get)

        # Make a new dict to avoid in-place modification while iterating
        qm = {key: _qm[key] - _qm[qm_minimum_index] for key in _qm}

        axis.plot(
            qm.keys(),
            qm.values(),
            "k.-",
            label=f"QM {molecule_id}",
        )

        for force_field in force_fields:
            mm = dict(sorted(store.get_mm_energies_by_molecule_id(molecule_id, force_field=force_field).items()))
            if len(mm) == 0:
                continue

            axis.plot(
                mm.keys(),
                [val - mm[qm_minimum_index] for val in mm.values()],
                "o--",
                label=force_field,
            )

        axis.legend(loc=0)
        axis.grid(axis="both")

    fig.savefig(f"{plot_dir}/random.png")

    plot_cdf(force_fields, output_metrics, plot_dir)


def plot_cdf(force_fields: list[str], metrics_file: str, plot_dir: str):
    metrics = MetricCollection.parse_file(metrics_file)

    x_ranges = {
        "rmsd": (0, 0.18),
        "rmse": (-0.3, 5),
    }

    rmsds = {
        force_field: {key: val.rmsd for key, val in metrics.metrics[force_field].items()}
        for force_field in metrics.metrics.keys()
    }

    rmses = {
        force_field: {key: val.rmse for key, val in metrics.metrics[force_field].items()}
        for force_field in metrics.metrics.keys()
    }

    data = {
        "rmsd": rmsds,
        "rmse": rmses,
    }
    for key in ["rmsd", "rmse"]:
        figure, axis = pyplot.subplots()

        for force_field in force_fields:
            if key == "dde":
                _data = np.array(
                    [*data[key][force_field].values()],
                    dtype=float,
                )

                counts, bins = np.histogram(
                    _data[np.isfinite(_data)],
                    bins=np.linspace(-15, 15, 31),
                )

                axis.stairs(counts, bins, label=force_field)

                axis.set_ylabel("Count")

            else:
                sorted_data = np.sort([*data[key][force_field].values()])

                axis.plot(
                    sorted_data,
                    np.arange(1, len(sorted_data) + 1) / len(sorted_data),
                    "-",
                    label=force_field,
                )

                axis.set_xlabel(key)
                axis.set_ylabel("CDF")

                axis.set_xlim(x_ranges[key])
                axis.set_ylim((-0.05, 1.05))

        axis.legend(loc=0)

        figure.savefig(f"{plot_dir}/{key}.png", dpi=300)


if __name__ == "__main__":
    # This setup is necessary for reasons that confused me - both setting it up in the __main__ block and calling
    # freeze_support(). This is probably not necessary after MoleculeStore.optimize_mm() is called, so you can load up
    # the same database for later analysis once the MM conformers are stored
    pyplot.style.use("ggplot")
    freeze_support()
    main()
