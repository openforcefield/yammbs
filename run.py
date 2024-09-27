import pathlib
from multiprocessing import freeze_support

from openff.qcsubmit.results import OptimizationResultCollection

from yammbs import MoleculeStore


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

    if pathlib.Path(f"{data}.sqlite").exists():
        store = MoleculeStore(f"{data}.sqlite")
    else:
        store = MoleculeStore.from_qcsubmit_collection(
            OptimizationResultCollection.parse_file(
                f"yammbs/_tests/data/01-processed-qm-{data}.json",
            ),
            database_name=f"{data}.sqlite",
        )

    for force_field in force_fields:
        # This is called within each analysis method, but short-circuiting within them. It's convenient to call it here
        # with the freeze_support setup so that later analysis methods can trust that the MM conformers are there
        store.optimize_mm(force_field=force_field)

    with open("out.json", "w") as f:
        f.write(store.get_outputs().json())


if __name__ == "__main__":
    # This setup is necessary for reasons that confused me - both setting it up in the __main__ block and calling
    # freeze_support(). This is probably not necessary after MoleculeStore.optimize_mm() is called, so you can load up
    # the same database for later analysis once the MM conformers are stored
    freeze_support()
    main()
