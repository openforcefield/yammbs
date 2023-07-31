from multiprocessing import freeze_support
from openff.qcsubmit.results import OptimizationResultCollection

from ibstore._store import MoleculeStore

def main():
    store = MoleculeStore.from_qcsubmit_collection(
        OptimizationResultCollection.parse_file(
            "ibstore/_tests/data/01-processed-qm-ch.json",
        ),
        # TODO: Don't assume this file doesn't already exist
        database_name="tmp.sqlite",
    )

    # This is called within each analysis method, but short-circuiting within them. It's convenient to call it here
    # with the freeze_support setup so that later analysis methods can trust that the MM conformers are there
    store.optimize_mm()

    store.get_dde().to_csv("dde.csv")
    store.get_rmsd().to_csv("rsmd.csv")

if __name__ == "__main__":
    # This setup is necessary for reasons that confused me - both setting it up in the __main__ block and calling
    # freeze_support(). This is probably not necessary after MoleculeStore.optimize_mm() is called, so you can load up
    # the same database for later analysis once the MM conformers are stored
    freeze_support()
    main()
