"""
filter_collection:
    file: full-optimization-benchmark-1.json
    filters:
      - invalid_cmiles
      - incomplete_record
      - connectivity_change
"""
import importlib
from multiprocessing import get_context

from openff.qcsubmit.results.filters import (
    CMILESResultFilter,
    ConnectivityFilter,
    RecordStatusFilter,
    SMILESFilter,
    UnperceivableStereoFilter,
)
from openff.toolkit.topology.molecule import Molecule, SmilesParsingError
from qcportal.models.records import RecordStatusEnum
from tqdm import tqdm

N_PROCESSES = 16


def _can_parameterize(smiles: str) -> tuple[str, bool]:
    try:
        for toolkit in GLOBAL_TOOLKIT_REGISTRY.registered_toolkits:
            if isinstance(toolkit, OpenEyeToolkitWrapper):
                continue

            GLOBAL_TOOLKIT_REGISTRY.deregister_toolkit(toolkit)

        molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        force_field = ForceField("openff-1.3.0.offxml")

        force_field.create_openmm_system(molecule.to_topology())

    except:
        return smiles, False

    return smiles, True


class InvalidCMILESFilter(CMILESResultFilter):
    def _filter_function(self, result) -> bool:
        try:
            Molecule.from_mapped_smiles(result.cmiles, allow_undefined_stereo=True)
        except (ValueError, SmilesParsingError):
            return False

        return True


_SIMPLE_FILTERS = {
    "invalid_cmiles": InvalidCMILESFilter(),
    "incomplete_record": RecordStatusFilter(status=RecordStatusEnum.complete),
    "connectivity_change": ConnectivityFilter(),
    "unambiguous_stereo": UnperceivableStereoFilter(),
}


def filter_collection(
    collection_type: str,
    file: str,
    filters: list[str],
    filtered_file: str,
):
    print("importing collection class")
    collection_class = getattr(
        importlib.import_module("openff.qcsubmit.results"),
        collection_type,
    )

    print("loading file")
    collection = collection_class.parse_file(file)

    print("filtering data")
    for filter in filters:
        if filter in {"can_parametrize"}:
            print("building molecules")
            _, molecules = zip(*collection.to_records())

            print("uniquifying molecules")
            unique_molecules = {
                molecule.to_smiles(isomeric=True, mapped=False)
                for molecule in molecules
            }

            print("running parametrization")
            with get_context("spawn").Pool(processes=N_PROCESSES) as pool:
                parametrizable_smiles = {
                    smiles
                    for smiles, should_retain in tqdm(
                        pool.imap_unordered(_can_parameterize, unique_molecules),
                        total=len(unique_molecules),
                    )
                    if should_retain
                }

            print(f"applying filter {filter}")
            collection.filter(SMILESFilter(smiles_to_include=[*parametrizable_smiles]))

        elif filter in _SIMPLE_FILTERS:
            print(f"applying filter {filter}")
            collection.filter(_SIMPLE_FILTERS[filter])
        else:
            print(f"filter {filter} not known")

    print("writing data")
    with open(filtered_file, "w") as _file:
        _file.write(collection.json())
