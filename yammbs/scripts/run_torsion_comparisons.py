import pathlib
from multiprocessing import freeze_support

from matplotlib import pyplot

from yammbs.torsion import TorsionStore
from yammbs.torsion.inputs import QCArchiveTorsionDataset
from yammbs.torsion.outputs import MetricCollection
import numpy as np


def main():
    force_fields = [
        "openff-1.0.0",
        # "openff-1.1.0",
        # "openff-1.3.0",
        "openff-2.0.0",
        "openff-2.1.0",
        "openff-2.2.1",
    ]

    if not pathlib.Path("torsion_drive_data.json").exists():
        from openff.qcsubmit.results import TorsionDriveResultCollection

        collection = TorsionDriveResultCollection.parse_file("filtered-supp-td.json")

        with open("torsiondrive-data.json", "w") as f:
            f.write(QCArchiveTorsionDataset.from_qcsubmit_collection(collection).model_dump_json())

    dataset = QCArchiveTorsionDataset.model_validate_json(open("torsion_drive_data.json").read())

    if pathlib.Path("torsion-example.sqlite").exists():
        store = TorsionStore("torsion-example.sqlite")
    else:
        store = TorsionStore.from_torsion_dataset(
            dataset,
            database_name="torsion-example.sqlite",
        )

    for force_field in force_fields:
        store.optimize_mm(force_field=force_field, n_processes=24)

    if not pathlib.Path("minimized.json").exists():
        with open("minimized.json", "w") as f:
            f.write(store.get_outputs().model_dump_json())

    if not pathlib.Path("metrics.json").exists():
        with open("metrics.json", "w") as f:
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

    fig.savefig("random.png")

    plot_cdf(force_fields)

def plot_cdf(force_fields: list[str]):
    metrics = MetricCollection.parse_file("metrics.json")

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

        figure.savefig(f"{key}.png", dpi=300)


if __name__ == "__main__":
    # This setup is necessary for reasons that confused me - both setting it up in the __main__ block and calling
    # freeze_support(). This is probably not necessary after MoleculeStore.optimize_mm() is called, so you can load up
    # the same database for later analysis once the MM conformers are stored
    freeze_support()
    main()
