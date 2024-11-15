import logging
import pathlib
from contextlib import contextmanager
from typing import Generator, Iterable

from numpy.typing import NDArray
from openff.qcsubmit.results import TorsionDriveResultCollection
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing_extensions import Self

from yammbs._molecule import _smiles_to_inchi_key
from yammbs._types import Pathlike
from yammbs.exceptions import DatabaseExistsError
from yammbs.torsion._db import (
    DBBase,
    DBMMTorsionPointRecord,
    DBQMTorsionPointRecord,
    DBTorsionRecord,
)
from yammbs.torsion._session import TorsionDBSessionManager
from yammbs.torsion.inputs import QCArchiveTorsionDataset
from yammbs.torsion.models import MMTorsionPointRecord, QMTorsionPointRecord, TorsionRecord

LOGGER = logging.getLogger(__name__)


class TorsionStore:
    def __len__(self):
        with self._get_session() as db:
            return db.db.query(DBTorsionRecord.mapped_smiles).count()

    def __init__(self, database_path: Pathlike = "torsion-store.sqlite"):
        database_path = pathlib.Path(database_path)

        if not database_path.suffix.lower() == ".sqlite":
            raise NotImplementedError(
                f"Only paths to SQLite databases ending in .sqlite are supported. Given: {database_path}",
            )

        self.database_url = f"sqlite:///{database_path.resolve()}"
        self.engine = create_engine(self.database_url)
        DBBase.metadata.create_all(self.engine)

        self._sessionmaker = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        with self._get_session() as db:
            assert db is not None
            # self.db_version = db.check_version()
            # self.general_provenance = db.get_general_provenance()
            # self.software_provenance = db.get_software_provenance()

    @contextmanager
    def _get_session(self) -> Generator[TorsionDBSessionManager, None, None]:
        session = self._sessionmaker()
        try:
            yield TorsionDBSessionManager(session)
            session.commit()
        except BaseException as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def store_torsion_record(
        self,
        records: TorsionRecord | Iterable[TorsionRecord],
    ):
        """Store molecules and their computed properties in the data store."""
        if isinstance(records, TorsionRecord):
            records = [records]

        with self._get_session() as db:
            for record in records:
                db.store_torsion_record(record)

    def store_qm_point(
        self,
        point: QMTorsionPointRecord,
    ):
        with self._get_session() as db:
            db.store_qm_torsion_point(point)

    def store_mm_point(
        self,
        point: MMTorsionPointRecord,
    ):
        with self._get_session() as db:
            db.store_mm_torsion_point(point)

    def get_molecule_ids(self) -> list[int]:
        """
        Get the molecule IDs of all records in the store.

        These are likely to be integers sequentially incrementing from 1, but that
        is not guaranteed.
        """
        with self._get_session() as db:
            return [molecule_id for (molecule_id,) in db.db.query(DBTorsionRecord.id).distinct()]

    # TODO: Allow by multiple selectors (smiles: list[str])
    def get_molecule_id_by_smiles(self, smiles: str) -> int:
        with self._get_session() as db:
            return next(id for (id,) in db.db.query(DBTorsionRecord.id).filter_by(mapped_smiles=smiles).all())

    # TODO: Allow by multiple selectors (id: list[int])
    def get_smiles_by_molecule_id(self, id: int) -> str:
        with self._get_session() as db:
            return next(smiles for (smiles,) in db.db.query(DBTorsionRecord.mapped_smiles).filter_by(id=id).all())

    def get_dihedral_indices_by_molecule_id(self, id: int) -> list[int]:
        with self._get_session() as db:
            return next(
                dihedral_indices
                for (dihedral_indices,) in db.db.query(DBTorsionRecord.dihedral_indices).filter_by(id=id).all()
            )

    def get_qm_points_by_molecule_id(self, id: int) -> dict[float, NDArray]:
        with self._get_session() as db:
            return {
                grid_id: coordinates
                for (grid_id, coordinates) in db.db.query(
                    DBQMTorsionPointRecord.grid_id,
                    DBQMTorsionPointRecord.coordinates,
                )
                .filter_by(parent_id=id)
                .all()
            }

    def get_mm_points_by_molecule_id(
        self,
        id: int,
        force_field: str,
    ) -> dict[float, NDArray]:
        with self._get_session() as db:
            return {
                grid_id: coordinates
                for (grid_id, coordinates) in db.db.query(
                    DBMMTorsionPointRecord.grid_id,
                    DBMMTorsionPointRecord.coordinates,
                )
                .filter_by(parent_id=id)
                .filter_by(force_field=force_field)
                .all()
            }

    def get_qm_energies_by_molecule_id(self, id: int) -> dict[float, float]:
        with self._get_session() as db:
            return {
                grid_id: energy
                for (grid_id, energy) in db.db.query(
                    DBQMTorsionPointRecord.grid_id,
                    DBQMTorsionPointRecord.energy,
                )
                .filter_by(parent_id=id)
                .all()
            }

    def get_mm_energies_by_molecule_id(self, id: int, force_field: str) -> dict[float, float]:
        with self._get_session() as db:
            return {
                grid_id: energy
                for (grid_id, energy) in db.db.query(
                    DBMMTorsionPointRecord.grid_id,
                    DBMMTorsionPointRecord.energy,
                )
                .filter_by(parent_id=id)
                .filter_by(force_field=force_field)
                .all()
            }

    @classmethod
    def from_torsion_dataset(
        cls,
        dataset: QCArchiveTorsionDataset,
        database_name: str,
    ) -> Self:
        if pathlib.Path(database_name).exists():
            raise DatabaseExistsError(f"Database {database_name} already exists.")

        store = cls(database_name)

        for qm_torsion in dataset.qm_torsions:
            torsion_record = TorsionRecord(
                mapped_smiles=qm_torsion.mapped_smiles,
                inchi_key=_smiles_to_inchi_key(qm_torsion.mapped_smiles),
                dihedral_indices=qm_torsion.dihedral_indices,
            )

            store.store_torsion_record(torsion_record)

            for angle in qm_torsion.coordinates:
                qm_point_record = QMTorsionPointRecord(
                    molecule_id=store.get_molecule_id_by_smiles(
                        torsion_record.mapped_smiles,
                    ),
                    grid_id=angle,  # TODO: This needs to be a tuple later
                    coordinates=qm_torsion.coordinates[angle],
                    energy=qm_torsion.energies[angle],
                )

                store.store_qm_point(qm_point_record)

        return store

    @classmethod
    def from_qcsubmit_collection(
        cls,
        collection: TorsionDriveResultCollection,
        database_name: str,
    ):
        """Convert a QCSubmit collection to TorsionDataset and use it to create a TorsionStore."""
        return cls.from_qm_dataset(
            dataset=QCArchiveTorsionDataset.from_qcsubmit_collection(collection),
            database_name=database_name,
        )

    def optimize_mm(
        self,
        force_field: str,
        n_processes: int = 2,
        chunksize: int = 32,
    ):
        """Run a constrained minimization of all torsion points."""
        # TODO: Pass through more options for constrained minimization process?

        from yammbs.torsion._minimize import _minimize_torsions

        for molecule_id in self.get_molecule_ids():
            with self._get_session() as db:
                # TODO: Implement "seen" behavior to short-circuit already-optimized torsions
                qm_data = tuple(
                    (grid_id, coordinates, energy)
                    for (grid_id, coordinates, energy) in db.db.query(
                        DBQMTorsionPointRecord.grid_id,
                        DBQMTorsionPointRecord.coordinates,
                        DBQMTorsionPointRecord.energy,
                    )
                    .filter_by(parent_id=molecule_id)
                    .all()
                )

            minimization_results = _minimize_torsions(
                mapped_smiles=self.get_smiles_by_molecule_id(molecule_id),
                dihedral_indices=self.get_dihedral_indices_by_molecule_id(molecule_id),
                qm_data=qm_data,
                force_field=force_field,
                n_processes=n_processes,
            )

            with self._get_session() as db:
                for result in minimization_results:
                    db.store_mm_torsion_point(
                        MMTorsionPointRecord(
                            molecule_id=molecule_id,
                            grid_id=result.grid_id,
                            coordinates=result.coordinates,
                            force_field=result.force_field,
                            energy=result.energy,
                        ),
                    )
