"""Analyse torsion drive data using different force fields."""

import pathlib
from multiprocessing import freeze_support

import click
import numpy as np
from matplotlib import pyplot
from openff.toolkit import Molecule
from rdkit.Chem import AllChem, Draw

from yammbs.torsion import TorsionStore
from yammbs.torsion.inputs import QCArchiveTorsionDataset
from yammbs.torsion.outputs import MetricCollection

pyplot.style.use("ggplot")


@click.command()
@click.option(
    "--base-force-fields",
    "-bf",
    multiple=True,
    default=[
        "openff-1.0.0",
        "openff-2.0.0",
        "openff-2.1.0",
        "openff-2.2.0",
        "openff-2.2.1",
    ],
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
    """Run torsion drive comparisons using specified force fields and input data."""
    force_fields = base_force_fields + extra_force_fields

    with open(qcarchive_torsion_data) as f:
        dataset = QCArchiveTorsionDataset.model_validate_json(f.read())

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
    plot_rms_stats(force_fields, output_metrics, plot_dir)
    plot_mean_error_distribution(force_fields, output_metrics, plot_dir)
    plot_rms_js_distance(force_fields, output_metrics, plot_dir)


def get_torsion_image(torsion_id: int, store: TorsionStore) -> pyplot.Figure:
    """Plot the torsion image for a given molecule ID."""
    smiles = store.get_smiles_by_torsion_id(torsion_id)
    dihedral_indices = store.get_dihedral_indices_by_torsion_id(torsion_id)

    # Use the mapped SMILES to get the molecule
    mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
    if mol is None:
        raise ValueError(f"Could not convert SMILES to molecule: {smiles}")

    rdmol = mol.to_rdkit()

    # Draw in 2D - compute 2D coordinates
    AllChem.Compute2DCoords(rdmol)
    # Highlight the dihedral
    atom_indices = [
        dihedral_indices[0],
        dihedral_indices[1],
        dihedral_indices[2],
        dihedral_indices[3],
    ]
    bond_indices = [
        rdmol.GetBondBetweenAtoms(atom_indices[0], atom_indices[1]).GetIdx(),
        rdmol.GetBondBetweenAtoms(atom_indices[1], atom_indices[2]).GetIdx(),
        rdmol.GetBondBetweenAtoms(atom_indices[2], atom_indices[3]).GetIdx(),
    ]
    img = Draw.MolToImage(
        rdmol,
        size=(300, 300),
        kekulize=True,
        wedgeBonds=True,
        highlightAtoms=atom_indices,
        highlightBonds=bond_indices,
    )
    # img = Draw.MolToImage(rdmol, size=(300, 300), kekulize=True, wedgeBonds=True)

    # Return the image so that it can be added to a matplotlib figure
    return img


def plot_torsions(plot_dir: str, force_fields: list[str], store: TorsionStore) -> None:
    """Plot the torsional energies for each molecule in the dataset."""
    n_rows = 8
    n_cols = 5

    # Adjust number of rows and columns down if we have fewer than 40 molecules
    n_molecules = len(store.get_torsion_ids())
    if n_molecules * 2 < n_rows * n_cols:
        n_rows = n_molecules // n_cols
        if n_molecules % n_cols != 0:
            n_rows += 1
    n_rows *= 2  # Two rows for each molecule

    n_torsions = n_rows * n_cols / 2  # Half the axes are for images

    fig, axes = pyplot.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))

    for i, torsion_id in enumerate(store.get_torsion_ids()):
        # Draw the molecule on the upper axis and the torsion plot on the lower axis
        if i >= n_torsions:
            break

        # Put the image on upper rows and the torsion plots underneath
        col = i % n_cols
        row = i // n_cols
        image_axis = axes[row * 2, col]
        torsion_axis = axes[row * 2 + 1, col]

        # Draw the molecule
        image_axis.imshow(get_torsion_image(torsion_id, store))
        image_axis.axis("off")

        # Plot the torsion data
        torsion_axis.set_title(f"ID: {torsion_id}")
        _qm = store.get_qm_energies_by_torsion_id(torsion_id)

        _qm = dict(sorted(_qm.items()))

        qm_minimum_index = min(_qm, key=_qm.get)

        # Make a new dict to avoid in-place modification while iterating
        qm = {key: _qm[key] - _qm[qm_minimum_index] for key in _qm}

        # Assume a default grid spacing of 15 degrees (BespokeFit default)
        angles = np.arange(-165, 195, 15)
        assert len(angles) == len(qm), "QM data and angles should match in length"

        torsion_axis.plot(
            angles,
            qm.values(),
            "k.-",
            label="QM",
        )

        for force_field in force_fields:
            mm = dict(sorted(store.get_mm_energies_by_torsion_id(torsion_id, force_field=force_field).items()))
            if len(mm) == 0:
                continue

            torsion_axis.plot(
                angles,
                [val - mm[qm_minimum_index] for val in mm.values()],
                "o--",
                label=force_field,
            )

        # Only add the axis if this is the last in the row - and add it off to the right
        if col == n_cols - 1:
            torsion_axis.legend(loc=0, bbox_to_anchor=(1.05, 1), borderaxespad=0)

        # Label the axes
        torsion_axis.set_ylabel(r"Energy / kcal mol$^{-1}$")
        torsion_axis.set_xlabel("Torsion angle / degrees")

    # Hide any unused axes
    for i in range(n_molecules, n_rows * n_cols):
        if i >= n_torsions:
            break
        col = i % n_cols
        row = i // n_cols
        axes[row * 2, col].axis("off")
        axes[row * 2 + 1, col].axis("off")

    fig.tight_layout()
    fig.savefig(f"{plot_dir}/torsions.png", dpi=300, bbox_inches="tight")


def plot_cdfs(force_fields: list[str], metrics_file: str, plot_dir: str):
    """Plot the cumulative distribution functions for the RMSD, RMSE, and Jensen-Shannon distance."""
    metrics = MetricCollection.parse_file(metrics_file)

    x_ranges = {"rmsd": (0, 0.14), "rmse": (-0.3, 5), "js_distance": (None, None)}

    units = {
        "rmsd": r"$\mathrm{\AA}$",
        "rmse": r"kcal mol$^{-1}$",
        "js_distance": "",
    }

    rmsds = {
        force_field: {key: val.rmsd for key, val in metrics.metrics[force_field].items()}
        for force_field in metrics.metrics.keys()
    }

    rmses = {
        force_field: {key: val.rmse for key, val in metrics.metrics[force_field].items()}
        for force_field in metrics.metrics.keys()
    }

    js_dists = {
        force_field: {key: val.js_distance[0] for key, val in metrics.metrics[force_field].items()}
        for force_field in metrics.metrics.keys()
    }

    js_div_temp = list(list(metrics.metrics.values())[0].values())[0].js_distance[1]

    data = {
        "rmsd": rmsds,
        "rmse": rmses,
        "js_distance": js_dists,
    }
    for key in ["rmsd", "rmse", "js_distance"]:
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

                x_label = (
                    key.upper() + " / " + units[key]
                    if key != "js_distance"
                    else f"Jensen-Shannon Distance at {js_div_temp} K"
                )
                axis.set_xlabel(x_label)
                axis.set_ylabel("CDF")

                axis.set_xlim(x_ranges[key])
                axis.set_ylim((-0.05, 1.05))

        axis.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        figure.savefig(f"{plot_dir}/{key}.png", dpi=300, bbox_inches="tight")


def get_rms(array: np.ndarray) -> float:
    """Calculate the root mean square of an array."""
    return np.sqrt(np.mean(array**2))


def plot_rms_stats(
    force_fields: list[str],
    metrics_file: str,
    plot_dir: str,
) -> None:
    """Plot the RMS values for the RMSD and RMSE."""
    metrics = MetricCollection.parse_file(metrics_file)

    units = {
        "rmsd": r"$\mathrm{\AA}$",
        "rmse": r"kcal mol$^{-1}$",
    }

    rms_rmses = {
        force_field: get_rms(np.array([val.rmse for val in metrics.metrics[force_field].values()]))
        for force_field in force_fields
    }

    rms_rmsds = {
        force_field: get_rms(np.array([val.rmsd for val in metrics.metrics[force_field].values()]))
        for force_field in force_fields
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


def plot_rms_js_distance(
    force_fields: list[str],
    metrics_file: str,
    plot_dir: str,
) -> None:
    """Plot the RMS JS distance for each force field."""
    metrics = MetricCollection.parse_file(metrics_file)

    rms_js_distance = {
        force_field: get_rms(np.array([val.js_distance[0] for val in metrics.metrics[force_field].values()]))
        for force_field in force_fields
    }

    js_div_temp = list(list(metrics.metrics.values())[0].values())[0].js_distance[1]

    # Plot mean JS distance
    figure, axis = pyplot.subplots()

    axis.bar(rms_js_distance.keys(), rms_js_distance.values(), color=pyplot.cm.tab10.colors)
    axis.set_ylabel(f"Mean Jensen-Shannon Distance at {js_div_temp} K")

    # Set x-ticks to be vertical
    pyplot.xticks(rotation=90)

    # Save the figure
    figure.tight_layout()
    figure.savefig(f"{plot_dir}/mean_js_distance.png", dpi=300, bbox_inches="tight")


def plot_mean_error_distribution(
    force_fields: list[str],
    metrics_file: str,
    plot_dir: str,
) -> None:
    """Plot the distribution of mean errors for each force field."""
    metrics = MetricCollection.parse_file(metrics_file)

    units = {
        "mean_error": r"kcal mol$^{-1}$",
    }

    mean_errors = {
        force_field: np.array([val.mean_error for val in metrics.metrics[force_field].values()])
        for force_field in force_fields
    }
    # Plot mean error distribution using kernel density estimation
    figure, axis = pyplot.subplots(figsize=(10, 4))
    import seaborn as sns

    for force_field in mean_errors.keys():
        sns.kdeplot(
            data=mean_errors[force_field],
            label=force_field,
            ax=axis,
        )
    axis.set_xlabel("Mean Error / " + units["mean_error"])
    axis.set_ylabel("Density")
    axis.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Save the figure
    figure.tight_layout()
    figure.savefig(f"{plot_dir}/mean_error_distribution.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # This setup is necessary for reasons that confused me - both setting it up in the __main__ block and calling
    # freeze_support(). This is probably not necessary after MoleculeStore.optimize_mm() is called, so you can load up
    # the same database for later analysis once the MM conformers are stored
    freeze_support()
    main()
