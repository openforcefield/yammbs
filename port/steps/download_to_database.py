import importlib
import logging
import pathlib
from typing import Iterable

import qcelemental
from ibstore._store import MoleculeStore
from ibstore.record import MoleculeRecord
from openff.toolkit import Molecule
from qcportal import FractalClient

hartree2kcalmol = qcelemental.constants.hartree2kcalmol


def download_to_database(
    collection_type: str,
    data_set: str,
    database: str,
    filter_smiles: Iterable[str] = tuple(),
    spec_name: str = "default",
):
    if pathlib.Path(database).exists():
        return

    logging.info("starting client")
    client = FractalClient()

    logging.info("importing collection class")
    collection_class = getattr(
        importlib.import_module("openff.qcsubmit.results"),
        collection_type,
    )

    logging.info("retrieving data")
    data = collection_class.from_server(
        client=client,
        datasets=data_set,
        spec_name="default",
    )

    logging.info("filtering data")
    ind_to_del = []
    for i, item in enumerate(data.entries["https://api.qcarchive.molssi.org:443/"]):
        for smiles in filter_smiles:
            if smiles.upper() in item.cmiles or smiles.lower() in item.cmiles:
                ind_to_del.append(i)
                continue

    for ind in sorted(ind_to_del, reverse=True):
        print("deleting implicit hydrogen entry: ", ind)
        del data.entries["https://api.qcarchive.molssi.org:443/"][ind]

    logging.info("creating records")
    records = list()
    for record_and_molecule in data.to_records():
        record = record_and_molecule[0]
        molecule: Molecule = record_and_molecule[1]

        if record.status != "COMPLETE":
            continue

        records.append(
            MoleculeRecord.from_qcsubmit_record(
                qcarchive_id=record.id,
                qcarchive_energy=record.get_final_energy() * hartree2kcalmol,
                molecule=molecule,
            )
        )

    logging.info("storing records")
    MoleculeStore(database).store(records)
