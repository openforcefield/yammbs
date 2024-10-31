from yammbs._session import DBSessionManager
from yammbs.torsion._db import DBMMTorsionPointRecord, DBQMTorsionPointRecord
from yammbs.torsion.models import MMTorsionPointRecord, QMTorsionPointRecord


# TODO: Composition over inheritance
class TorsionDBSessionManager(DBSessionManager):
    pass

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
