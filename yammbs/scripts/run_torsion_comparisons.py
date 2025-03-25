import pathlib
from multiprocessing import freeze_support

import click
import numpy as np
from matplotlib import pyplot

from yammbs.torsion import TorsionStore
from yammbs.torsion.inputs import QCArchiveTorsionDataset
from yammbs.torsion.outputs import MetricCollection

pyplot.style.use("ggplot")


@click.command()
@click.option(
    "--base-force-fields",
    "-bf",
    multiple=True,
    default=["openff-1.0.0", "openff-2.0.0", "openff-2.1.0", "openff-2.2.1"],
    help="List of force fields to use for optimization.",
)
@click.option(
    "--extra-force-fields",
    "-ef",
    multiple=True,
    default=[],
    help="Extra (local) force fields to use for optimization.",
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
    base_force_fields: list[str],
    extra_force_fields: list[str],
    qcarchive_torsion_data: str,
    database_file: str,
    output_metrics: str,
    output_minimized: str,
    plot_dir: str,
) -> None:
    """
    Run torsion drive comparisons using specified force fields and input data.
    """
    force_fields = base_force_fields + extra_force_fields

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

    # Plot!
    plot_torsions(plot_dir, force_fields, store)
    plot_cdfs(force_fields, output_metrics, plot_dir)
    plot_rms_stats(output_metrics, plot_dir)


def plot_torsions(plot_dir: str, force_fields: list[str], store: TorsionStore) -> None:
    fig, axes = pyplot.subplots(5, 4, figsize=(20, 20))

    for molecule_id, axis in zip(store.get_molecule_ids(), axes.flatten()):
        _qm = store.get_qm_energies_by_molecule_id(molecule_id)

        _qm = dict(sorted(_qm.items()))

        qm_minimum_index = min(_qm, key=_qm.get)

        # Make a new dict to avoid in-place modification while iterating
        qm = {key: _qm[key] - _qm[qm_minimum_index] for key in _qm}

        # Assume a default grid spacing of 15 degrees (BespokeFit default)
        angles = np.arange(-165, 195, 15)
        assert len(angles) == len(qm), "QM data and angles should match in length"

        axis.plot(
            angles,
            qm.values(),
            "k.-",
            label=f"QM {molecule_id}",
        )

        for force_field in force_fields:
            mm = dict(sorted(store.get_mm_energies_by_molecule_id(molecule_id, force_field=force_field).items()))
            if len(mm) == 0:
                continue

            axis.plot(
                angles,
                [val - mm[qm_minimum_index] for val in mm.values()],
                "o--",
                label=force_field,
            )

        axis.legend(loc=0)

        # Label the axes
        axis.set_ylabel(r"Energy / kcal mol$^{-1}$")
        axis.set_xlabel("Torsion angle / degrees")

    fig.savefig(f"{plot_dir}/random.png")


def plot_cdfs(force_fields: list[str], metrics_file: str, plot_dir: str):
    metrics = MetricCollection.parse_file(metrics_file)

    x_ranges = {
        "rmsd": (0, 0.14),
        "rmse": (-0.3, 5),
    }

    units = {
        "rmsd": r"$\mathrm{\AA}$",
        "rmse": r"kcal mol$^{-1}$",
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

                axis.set_xlabel(key.upper() + " / " + units[key])
                axis.set_ylabel("CDF")

                axis.set_xlim(x_ranges[key])
                axis.set_ylim((-0.05, 1.05))

        axis.legend(loc=0)

        figure.savefig(f"{plot_dir}/{key}.png", dpi=300)


def get_rms(array: np.ndarray) -> float:
    """
    Calculate the root mean square of an array.
    """
    return np.sqrt(np.mean(array**2))


def plot_rms_stats(
    metrics_file: str,
    plot_dir: str,
) -> None:
    metrics = MetricCollection.parse_file(metrics_file)

    units = {
        "rmsd": r"$\mathrm{\AA}$",
        "rmse": r"kcal mol$^{-1}$",
    }

    rms_rmses = {
        force_field: get_rms(np.array([val.rmse for val in metrics.metrics[force_field].values()]))
        for force_field in metrics.metrics.keys()
    }

    rms_rmsds = {
        force_field: get_rms(np.array([val.rmsd for val in metrics.metrics[force_field].values()]))
        for force_field in metrics.metrics.keys()
    }

    # Plot RMS values
    for key, data in zip(["rmsd", "rmse"], [rms_rmsds, rms_rmses]):
        figure, axis = pyplot.subplots()

        # Use different colors for each bar - the same as for the CDFs
        axis.bar(data.keys(), data.values(), color=pyplot.cm.tab10.colors)
        axis.set_ylabel(key.upper() + " / " + units[key])

        # Set x-ticks to be vertical
        pyplot.xticks(rotation=90)

        # Save the figure
        figure.tight_layout()
        figure.savefig(f"{plot_dir}/{key}_rms.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # This setup is necessary for reasons that confused me - both setting it up in the __main__ block and calling
    # freeze_support(). This is probably not necessary after MoleculeStore.optimize_mm() is called, so you can load up
    # the same database for later analysis once the MM conformers are stored
    freeze_support()
    main()
