import importlib
from collections import defaultdict

import qcelemental
from openeye import oechem


def _process_molecule(record_and_molecule) -> oechem.OEMol:
    """Convert a QC record and its associated molecule into an OE molecule which
    has been tagged with the associated SMILES, final energy and record id."""

    record, molecule = record_and_molecule

    oe_molecule = molecule.to_openeye()
    oechem.OE3DToInternalStereo(oe_molecule)

    final_energy = record.get_final_energy() * qcelemental.constants.hartree2kcalmol

    # add name and energy tag to the mol
    oechem.OESetSDData(oe_molecule, "SMILES QCArchive", molecule.to_smiles())
    oechem.OESetSDData(oe_molecule, "Energy QCArchive", str(final_energy))
    oechem.OESetSDData(oe_molecule, "Record QCArchive", str(record.id))

    return oe_molecule


def group_to_sdf(
    collection_type: str,
    file: str,
    sdf_file: str,
):
    print("importing collection class")
    collection_class = getattr(
        importlib.import_module("openff.qcsubmit.results"),
        collection_type,
    )

    print("loading file")
    collection = collection_class.parse_file(file)

    print("building molecules")
    records_and_molecules = collection.to_records()

    grouped_molecules = defaultdict(list)

    for record, molecule in records_and_molecules:
        molecule = molecule.canonical_order_atoms()

        smiles = molecule.to_smiles(isomeric=False, explicit_hydrogens=True)
        grouped_molecules[smiles].append((record, molecule))

    print("processing molecules")
    processed_oe_molecules = [
        _process_molecule(record_and_molecule)
        for record_and_molecule in records_and_molecules
    ]

    print("writing molecules to sdf")

    output_steam = oechem.oemolostream(sdf_file)

    final_record_ids = set()

    for i, oe_molecule in enumerate(processed_oe_molecules):
        final_record_ids.add(oechem.OEGetSDData(oe_molecule, "Record QCArchive"))

        oe_molecule.SetTitle(f"full_{i + 1}")
        oechem.OEWriteConstMolecule(output_steam, oe_molecule)

    output_steam.close()
