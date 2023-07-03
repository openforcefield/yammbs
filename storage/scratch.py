import qcportal
from ibstore._store import MoleculeStore
from ibstore.record import MoleculeRecord
from openff.qcsubmit.results import OptimizationResultCollection
from openff.toolkit import Molecule
from qcelemental.constants import hartree2kcalmol

client = qcportal.FractalClient()

dataset = OptimizationResultCollection.from_server(
    client,
    "OpenFF Gen 2 Opt Set 3 Pfizer Discrepancy",
    spec_name="default",
)

records = list()
for record_and_molecule in dataset.to_records():
    record = record_and_molecule[0]
    molecule: Molecule = record_and_molecule[1]

    if record.status != "COMPLETE":
        continue

    records.append(
        MoleculeRecord.from_qcsubmit_record(
            qcarchive_id=record.id,
            molecule=molecule,
            qcarchive_energy=record.get_final_energy() * hartree2kcalmol,
        )
    )

MoleculeStore("dataset.sqlite").store(records)
