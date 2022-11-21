import pathlib

from openff.toolkit.topology.molecule import Molecule
from openff.units import unit

from ib.compute.minimize import _minimize
from ib.core.conformers import _calc_rmsd


def _run_rmsd(
    offxml: str,
    molecule: str,
    outfile: str,
):
    sdf_path = pathlib.Path(molecule)

    rmsd: dict[str, unit.Quantity] = dict()

    for sdf_file in sdf_path.glob("*.sdf"):

        molecule = Molecule.from_file(sdf_file.as_posix(), "sdf")

        rmsd[sdf_file.stem] = _calc_rmsd(molecule, _minimize(molecule, offxml))

        with open(outfile, "w") as out_file:

            out_file.write("id,rmsd\n")

            for key, value in rmsd.items():
                out_file.write(f"{key},{value.m_as(unit.angstrom)}\n")
