import json
from dataclasses import dataclass

import numpy
from openff.qcsubmit.results import OptimizationResultCollection


@dataclass
class CachedResult:
    """All of the fields necessary to emulate
    MoleculeStore.from_qcsubmit_collection without calling
    OptimizationResultCollection.to_records

    """

    mapped_smiles: str
    inchi_key: str
    coordinates: numpy.ndarray
    qc_record_id: int
    qc_record_final_energy: float

    def to_dict(self):
        return dict(
            mapped_smiles=self.mapped_smiles,
            inchi_key=self.inchi_key,
            coordinates=self.coordinates.tolist(),
            qc_record_id=self.qc_record_id,
            qc_record_final_energy=self.qc_record_final_energy,
        )


class CachedResultCollection:
    inner: list[CachedResult]

    def __init__(self):
        self.inner = []

    @classmethod
    def from_qcsubmit_collection(
        cls,
        collection: OptimizationResultCollection,
    ):
        import qcelemental
        from tqdm import tqdm

        hartree2kcalmol = qcelemental.constants.hartree2kcalmol

        ret = cls()
        for qcarchive_record, molecule in tqdm(
            collection.to_records(),
            desc="Converting records to molecules",
        ):
            energy = qcarchive_record.energies[-1] * hartree2kcalmol
            ret.inner.append(
                CachedResult(
                    mapped_smiles=molecule.to_smiles(
                        mapped=True,
                        isomeric=True,
                    ),
                    inchi_key=molecule.to_inchi(fixed_hydrogens=True),
                    coordinates=molecule.conformers[0],
                    qc_record_id=qcarchive_record.id,
                    qc_record_final_energy=energy,
                ),
            )
        return ret

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.inner, default=CachedResult.to_dict, **kwargs)

    @classmethod
    def from_json(cls, filename):
        """Load a `CachedResultCollection` from `filename`"""
        with open(filename) as inp:
            data = json.load(inp)

        ret = cls()
        for entry in data:
            coordinates = numpy.array(entry["coordinates"]).reshape(-1, 3)
            ret.inner.append(
                CachedResult(
                    mapped_smiles=entry["mapped_smiles"],
                    inchi_key=entry["inchi_key"],
                    coordinates=coordinates,
                    qc_record_id=entry["qc_record_id"],
                    qc_record_final_energy=entry["qc_record_final_energy"],
                ),
            )
        return ret
