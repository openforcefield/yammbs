import logging
import math
import pathlib
from contextlib import contextmanager
from typing import Generator, Iterable

import numpy
from numpy.typing import NDArray
from openff.qcsubmit.results import TorsionDriveResultCollection
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing_extensions import Self

from yammbs._molecule import _smiles_to_inchi_key
from yammbs._types import Pathlike
from yammbs.analysis import get_rmsd
from yammbs.exceptions import DatabaseExistsError
from yammbs.torsion._db import (
    DBBase,
    DBMMTorsionPointRecord,
    DBQMTorsionPointRecord,
    DBTorsionRecord,
)
from yammbs.torsion._session import TorsionDBSessionManager
from yammbs.torsion.analysis import RMSD, LogSSE, LogSSECollection, RMSDCollection, _normalize
from yammbs.torsion.inputs import QCArchiveTorsionDataset
from yammbs.torsion.models import MMTorsionPointRecord, QMTorsionPointRecord, TorsionRecord
from yammbs.torsion.outputs import Metric, MetricCollection, MinimizedTorsionDataset

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

    def get_force_fields(
        self,
    ) -> list[str]:
        """Return a list of all force fields with some torsion points stored."""
        with self._get_session() as db:
            return [
                force_field
                for (force_field,) in db.db.query(
                    DBMMTorsionPointRecord.force_field,
                ).distinct()
            ]

    def get_dihedral_indices_by_molecule_id(self, id: int) -> tuple[int, int, int, int]:
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
        return cls.from_torsion_dataset(
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

        id_to_smiles = {
            molecule_id: self.get_smiles_by_molecule_id(molecule_id) for molecule_id in self.get_molecule_ids()
        }
        id_to_dihedral_indices = {
            molecule_id: self.get_dihedral_indices_by_molecule_id(molecule_id)
            for molecule_id in self.get_molecule_ids()
        }

        with self._get_session() as db:
            # TODO: Implement "seen" behavior to short-circuit already-optimized torsions
            data: Generator[
                tuple[
                    int,
                    str,
                    tuple[int, int, int, int],
                    float,
                    NDArray,
                    float,
                ],
                None,
                None,
            ] = (  # Probably a better way to do this with some proper database query with join
                (
                    molecule_id,
                    id_to_smiles[molecule_id],
                    id_to_dihedral_indices[molecule_id],
                    grid_id,
                    coordinates,
                    energy,
                )
                for (molecule_id, grid_id, coordinates, energy) in db.db.query(
                    DBQMTorsionPointRecord.parent_id,
                    DBQMTorsionPointRecord.grid_id,
                    DBQMTorsionPointRecord.coordinates,
                    DBQMTorsionPointRecord.energy,
                ).all()
            )

        minimization_results = _minimize_torsions(
            data=data,
            force_field=force_field,
            n_processes=n_processes,
        )

        with self._get_session() as db:
            for result in minimization_results:
                db.store_mm_torsion_point(
                    MMTorsionPointRecord(
                        molecule_id=result.molecule_id,
                        grid_id=result.grid_id,
                        coordinates=result.coordinates,
                        force_field=result.force_field,
                        energy=result.energy,
                    ),
                )

    def get_log_sse(
        self,
        force_field: str,
        molecule_ids: list[int] | None = None,
        skip_check: bool = False,
    ) -> LogSSECollection:
        if not molecule_ids:
            molecule_ids = self.get_molecule_ids()

        if not skip_check:
            self.optimize_mm(force_field=force_field)

        log_sses = LogSSECollection()

        for molecule_id in self.get_molecule_ids():
            if molecule_id not in molecule_ids:
                continue

            qm, mm = _normalize(
                qm=self.get_qm_energies_by_molecule_id(molecule_id),
                mm=self.get_mm_energies_by_molecule_id(molecule_id, force_field),
            )

            log_sses.append(
                LogSSE(
                    id=molecule_id,
                    log_sse=math.log(sum([(mm[key] - qm[key]) ** 2 for key in qm])),
                ),
            )

        return log_sses

    def get_rmsd(
        self,
        force_field: str,
        molecule_ids: list[int] | None = None,
        skip_check: bool = False,
    ) -> RMSDCollection:
        """Get the RMSD summed over the torsion profile."""
        from openff.toolkit import Molecule

        if not molecule_ids:
            molecule_ids = self.get_molecule_ids()

        if not skip_check:
            self.optimize_mm(force_field=force_field)

        rmsds = RMSDCollection()

        for molecule_id in molecule_ids:
            if molecule_id not in molecule_ids:
                continue

            qm_points = self.get_qm_points_by_molecule_id(id=molecule_id)
            mm_points = self.get_mm_points_by_molecule_id(id=molecule_id, force_field=force_field)

            molecule = Molecule.from_mapped_smiles(
                self.get_smiles_by_molecule_id(molecule_id),
                allow_undefined_stereo=True,
            )

            tmp = [get_rmsd(molecule, qm_points[key], mm_points[key]) for key in qm_points]
            print(tmp)
            rmsds.append(
                RMSD(
                    id=molecule_id,
                    rmsd=sum(tmp),
                ),
            )

        return rmsds

    def get_outputs(self) -> MinimizedTorsionDataset:
        from yammbs.torsion.outputs import MinimizedTorsionProfile

        output_dataset = MinimizedTorsionDataset()

        with self._get_session() as db:
            for force_field in self.get_force_fields():
                output_dataset.mm_torsions[force_field] = list()

                for molecule_id in self.get_molecule_ids():
                    mm_data = tuple(
                        (grid_id, coordinates, energy)
                        for (grid_id, coordinates, energy) in db.db.query(
                            DBMMTorsionPointRecord.grid_id,
                            DBMMTorsionPointRecord.coordinates,
                            DBMMTorsionPointRecord.energy,
                        )
                        .filter_by(parent_id=molecule_id)
                        .filter_by(force_field=force_field)
                        .all()
                    )

                    if len(mm_data) == 0:
                        continue

                    # TODO: Call _normalize here?
                    output_dataset.mm_torsions[force_field].append(
                        MinimizedTorsionProfile(
                            mapped_smiles=self.get_smiles_by_molecule_id(molecule_id),
                            dihedral_indices=self.get_dihedral_indices_by_molecule_id(molecule_id),
                            coordinates={grid_id: coordinates for grid_id, coordinates, _ in mm_data},
                            energies={grid_id: energy for grid_id, _, energy in mm_data},
                        ),
                    )

        return output_dataset

    def get_metrics(
        self,
    ) -> MetricCollection:
        import pandas

        metrics = MetricCollection()

        # TODO: Optimize this for speed
        for force_field in self.get_force_fields():
            log_sses = self.get_log_sse(force_field=force_field, skip_check=True).to_dataframe()
            rmsds = self.get_rmsd(force_field=force_field, skip_check=True).to_dataframe()

            dataframe = log_sses.join(rmsds)

            dataframe = dataframe.replace({pandas.NA: numpy.nan})

            metrics.metrics[force_field] = {
                id: Metric(  # type: ignore[misc]
                    log_sse=row["log_sse"],
                    rmsd=row["rmsd"],
                )
                for id, row in dataframe.iterrows()
            }

        return metrics