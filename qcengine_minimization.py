import qcelemental
import qcengine
from openff.toolkit.topology.molecule import Molecule

glycylglycine = Molecule.from_smiles("NCC(=O)NCC(O)=O")

glycylglycine.generate_conformers(n_conformers=1)

atomic_input = qcelemental.models.OptimizationInput(
    initial_molecule=glycylglycine.to_qcschema(),
    input_specification={
        "driver": "gradient",
        "model": {
            "method": "openff-1.3.0",
            "basis": "smirnoff",
        },
    },
    keywords={
        "coordsys": "dlc",
        "enforce": 0,
        "epsilon": 1e-05,
        "reset": True,
        "qccnv": False,
        "molcnv": False,
        "check": 0,
        "trust": 0.1,
        "tmax": 0.3,
        "maxiter": 300,
        "convergence_set": "gau",
        "program": "openmm",
    },
)

return_value = qcengine.compute_procedure(
    input_data=atomic_input, procedure="geometric"
)

print(return_value.error.error_message)
