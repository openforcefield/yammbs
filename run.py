import pathlib

import matplotlib.pyplot as plt
from openff.toolkit import ForceField, Molecule
from openff.units import unit

from ib.core.conformers import (
    ConformerGenerationConfig,
    _calc_rmsd,
    _generate_conformers,
)
from ib.core.openmm import _minimize
from ib.forcefields import GAFFForceFieldProvider, SMIRNOFFForceFieldProvider
from ib.molecules import OpenFFMolecule, OpenFFSingleMoleculeTopologyProvider
from ib.systems import GAFFSystemProvider, SMIRNOFFSystemProvider

p = pathlib.Path("molecules/").mkdir(exist_ok=True)

SMILES = [
    "CCOCn1nccc1-c1ccc(Oc2ccc(O)cc2)cc1",
    "CCOC(=O)CCNC(=O)Nc1scc(C)c1C(=O)O",
    "CC(C)=CCN1CCC(N)CC1",
    "COC(=O)c1nc2ccc(C(=O)O)cc2n1CCCN",
    "CN(C)CCCN1c2ccccc2CCc2ccccc21",
    "CS(=O)(=O)c1cc(CN)ccn1",
    "Cc1cnc(N2CCCCC2)o1",
    "C1CCC(PC2CCCCC2)CC1",
    "COc1ccc(C(=O)C2CNC2)cc1",
    "CCC[N+](CCC)(CCC)CCC",
]

conformer_config = ConformerGenerationConfig()

sage = SMIRNOFFForceFieldProvider(
    identifier="openff-2.0.0", force_field=ForceField("openff-2.0.0.offxml")
)

rmsds = list()

for index, smiles in enumerate(SMILES):
    molecule = Molecule.from_smiles(smiles)

    molecule.to_file(f"molecules/{index:06d}.sdf", file_format="sdf")

    gaff = GAFFForceFieldProvider.from_molecule(molecule)
    print("generating conformers ...")
    for conformer_molecule in _generate_conformers(molecule, conformer_config):
        molecule_copy = Molecule(conformer_molecule)

        gaff_system = GAFFSystemProvider(
            identifier=gaff.identifier,
            topology=OpenFFSingleMoleculeTopologyProvider(
                components=[
                    OpenFFMolecule(
                        identifier="openff",
                        molecule=molecule_copy,
                    )
                ],
            ),
            force_field=gaff,
            positions=conformer_molecule.conformers[0],
        )

        gaff_molecule = Molecule(molecule_copy)
        gaff_molecule._conformers = [_minimize(gaff_system)]

        sage_system = SMIRNOFFSystemProvider(
            identifier="sage",
            topology=OpenFFSingleMoleculeTopologyProvider(
                components=[
                    OpenFFMolecule(
                        identifier="openff",
                        molecule=molecule_copy,
                    )
                ]
            ),
            force_field=sage,
            positions=conformer_molecule.conformers[0],
        )

        sage_molecule = Molecule(molecule_copy)
        sage_molecule._conformers = [_minimize(sage_system)]

        rmsd = _calc_rmsd(sage_molecule, gaff_molecule)
        rmsds.append(rmsd.m_as(unit.angstrom))

        print(round(rmsd, 3))

with open("rmsd.txt", "w") as file:
    for rmsd in rmsds:
        file.write(str(rmsd) + "\n")

plt.hist(rmsds)

plt.savefig("hist.pdf")
