from yammbs._session import DBSessionManager
from yammbs.torsion._db import DBQMTorsionPointRecord
from yammbs.torsion.models import QMTorsionPointRecord


# TODO: Composition over inheritance
class TorsionDBSessionmanager(DBSessionManager):
    pass

    def store_qm_torsion_point(self, record: QMTorsionPointRecord):
        self.db.add(
            DBQMTorsionPointRecord(
                parent_id=record.molecule_id,
                grid_id=record.grid_id,
                coordinates=record.coordinates,
            ),
        )
