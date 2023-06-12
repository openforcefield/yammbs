import json
import pathlib
from typing import Optional

import numpy
import qcelemental
import qcportal
import tqdm
from openff.toolkit import Molecule
from openff.units import unit
from qcportal import Molecule as QCMolecule

_hartree_avogadro = unit.hartree * unit.avogadro_constant


class ConnectivityRearrangedError(BaseException):
    pass


def _molecule_from_record(record) -> Optional[Molecule]:
    molecule = Molecule.from_qcschema(record)

    molecule._conformers = None

    return molecule


def _check_connectivity_rearrangements(molecule: Molecule):
    """
    https://github.com/openforcefield/openff-benchmark/blob/3b9ae9de3d19b744840c702155e8982a7df3e92e/openff/benchmark/geometry_optimizations/compute.py#L665

    Compare the connectivity implied by the molecule's geometry to that in its connectivity table.
    Returns True if the molecule's connectivity appears to have been rearranged.
    The method is taken from Josh Horton's work in QCSubmit
    https://github.com/openforcefield/openff-qcsubmit/blob/ce2df12d60ec01893e77cbccc50be9f0944a65db/openff/qcsubmit/results.py#L769
    """
    qc_molecule = molecule.to_qcschema()

    guessed_connectivity = qcelemental.molutil.guess_connectivity(
        qc_molecule.symbols, qc_molecule.geometry
    )

    if len(molecule.bonds) != len(guessed_connectivity):
        raise ConnectivityRearrangedError(
            "Number of bonds differs from guessed connectivity."
        )

    for bond in molecule.bonds:
        _bond = tuple([bond.atom1_index, bond.atom2_index])
        if _bond not in guessed_connectivity:
            if reversed(tuple(_bond)) not in guessed_connectivity:
                raise ConnectivityRearrangedError(
                    f"Bond {bond} not found in guessed connectivity."
                )


def _process_final_mol(
    output_id: str,
    molecule: Molecule,
    qcmolecule: QCMolecule,
    method: Optional[str] = None,
    basis: Optional[str] = None,
    program: Optional[str] = None,
    energies: list[float] = None,
) -> Molecule:
    qc_geometry = unit.Quantity(numpy.array(qcmolecule.geometry, float), unit.bohr)
    molecule._conformers = [qc_geometry.to(unit.angstrom)]

    _check_connectivity_rearrangements(molecule)

    molecule.name = output_id

    molecule.properties["method"] = method
    molecule.properties["basis"] = basis
    molecule.properties["program"] = program

    return molecule


def _write_molecule(molecule: Molecule, file_path: pathlib.Path):
    molecule.to_file(file_path.as_posix(), "SDF")


def _load_dataset_from_server(collection_name: str, client: qcportal.FractalClient):
    dataset = client.get_collection(
        "OptimizationDataset",
        collection_name,
    )

    dataset.status()

    with open("dataset.json", "w") as file:
        data = dataset.to_json()
        data.pop("history")

        json.dump(data, file)

    with open("dataset.json") as file:
        json.load(file)


def dataset_to_sdf(
    dataset: qcportal.collections.OptimizationDataset,
    output_path: pathlib.Path,
    delete_existing: bool = False,
):
    if not delete_existing:
        if pathlib.Path(output_path).exists():
            raise Exception("Output path already exists.")

    print("Running status ... ")
    dataset.status()
    print(" ... complete.")

    specifications = list(dataset.list_specifications().index)

    for specification in specifications:
        print(f"Starting specification {specification} ...")

        for _, record in tqdm.tqdm(dataset.df[specification].items()):
            print(f"Processing record {record} ...")

            index = record.final_molecule
            if index is None:
                continue

            outfile = output_path / specification / f"{index}.sdf"

            if outfile.exists():
                continue

            if not delete_existing:
                if pathlib.Path(outfile).exists():
                    continue

            molecule = _molecule_from_record(record.get_final_molecule())

            if molecule is None:
                continue

            final_molecule: QCMolecule = record.get_final_molecule()

            molecule = _process_final_mol(f"{index}", molecule, final_molecule)

            _write_molecule(molecule, outfile)

            print(f" ... finished {record}.")

        print(f" ... finished {specification}.")


if __name__ == "__main__":
    client = qcportal.FractalClient(verify=False)

    if not pathlib.Path("dataset.json").exists():
        print("Loading dataset from MolSSI server ...")

        _load_dataset_from_server(
            "OpenFF Iodine Chemistry Optimization Dataset v1.0",
            client,
        )

    print(" ... loaded.")

    print("Loading dataset from disk ...")
    dataset = qcportal.collections.OptimizationDataset.from_json(
        json.load(open("dataset.json")),
        client=client,
    )

    print(" ... loaded.")

    dataset_to_sdf(dataset, pathlib.Path("test"), True)
