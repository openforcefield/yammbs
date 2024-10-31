import logging
import pathlib
import random
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
from yammbs.models import MoleculeRecord
from yammbs.torsion._db import (
    DBBase,
    DBMMTorsionPointRecord,
    DBMoleculeRecord,
    DBQMTorsionPointRecord,
)
from yammbs.torsion._session import TorsionDBSessionManager
from yammbs.torsion.inputs import QCArchiveTorsionDataset
from yammbs.torsion.models import MMTorsionPointRecord, QMTorsionPointRecord

LOGGER = logging.getLogger(__name__)


class TorsionStore:
    def __len__(self):
        with self._get_session() as db:
            return db.db.query(DBMoleculeRecord.mapped_smiles).count()

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

    def store_molecule_record(
        self,
        records: MoleculeRecord | Iterable[MoleculeRecord],
    ):
        """Store molecules and their computed properties in the data store.

        Parameters
        ----------
        records: Iterable[MoleculeRecord]
            The QCArchive id and record of each molecule to store.
        """
        if isinstance(records, MoleculeRecord):
            records = [records]

        with self._get_session() as db:
            for record in records:
                db.store_molecule_record(record)

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

    # get_molecule_ids = MoleculeStore.get_molecule_ids

    def get_molecule_ids(self) -> list[int]:
        """
        Get the molecule IDs of all records in the store.

        These are likely to be integers sequentially incrementing from 1, but that
        is not guaranteed.
        """
        with self._get_session() as db:
            return [molecule_id for (molecule_id,) in db.db.query(DBMoleculeRecord.id).distinct()]

    # TODO: Allow by multiple selectors (smiles: list[str])
    def get_molecule_id_by_smiles(self, smiles: str) -> int:
        with self._get_session() as db:
            return next(id for (id,) in db.db.query(DBMoleculeRecord.id).filter_by(mapped_smiles=smiles).all())

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
            molecule_record = MoleculeRecord(
                mapped_smiles=qm_torsion.mapped_smiles,
                inchi_key=_smiles_to_inchi_key(qm_torsion.mapped_smiles),
            )

            store.store_molecule_record(molecule_record)

            for angle in qm_torsion.coordinates:
                qm_point_record = QMTorsionPointRecord(
                    molecule_id=store.get_molecule_id_by_smiles(
                        molecule_record.mapped_smiles,
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
        # TODO: Pass through options for constrained minimization process?

        # This is fake noise in place of data, just to test models
        for molecule_id in self.get_molecule_ids():
            with self._get_session() as db:
                qm_data = list(
                    (grid_id, coordinates, energy)
                    for (grid_id, coordinates, energy) in db.db.query(
                        DBQMTorsionPointRecord.grid_id,
                        DBQMTorsionPointRecord.coordinates,
                        DBQMTorsionPointRecord.energy,
                    )
                    .filter_by(parent_id=molecule_id)
                    .all()
                )

            with self._get_session() as db:
                for grid_id, coordinates, energy in qm_data:
                    db.store_mm_torsion_point(
                        MMTorsionPointRecord(
                            molecule_id=molecule_id,
                            grid_id=grid_id,
                            coordinates=coordinates,
                            force_field=force_field,
                            energy=energy + 2 * (random.random() - 0.5),
                        ),
                    )
