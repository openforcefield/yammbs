from yammbs._session import DBSessionManager
from yammbs.torsion._db import DBMMTorsionPointRecord, DBQMTorsionPointRecord, DBTorsionRecord
from yammbs.torsion.models import MMTorsionPointRecord, QMTorsionPointRecord, TorsionRecord


# TODO: Composition over inheritance
class TorsionDBSessionManager(DBSessionManager):
    def store_torsion_record(self, record: TorsionRecord):
        self.db.add(
            DBTorsionRecord(
                mapped_smiles=record.mapped_smiles,
                inchi_key=record.inchi_key,
                dihedral_indices=record.dihedral_indices,
            ),
        )

    def store_qm_torsion_point(self, record: QMTorsionPointRecord):
        self.db.add(
            DBQMTorsionPointRecord(
                parent_id=record.molecule_id,
                grid_id=record.grid_id,
                coordinates=record.coordinates,
                energy=record.energy,
            ),
        )

    def store_mm_torsion_point(self, record: MMTorsionPointRecord):
        self.db.add(
            DBMMTorsionPointRecord(
                parent_id=record.molecule_id,
                grid_id=record.grid_id,
                coordinates=record.coordinates,
                force_field=record.force_field,
                energy=record.energy,
            ),
        )
