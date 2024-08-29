import logging
import pathlib
from collections import defaultdict
from contextlib import contextmanager
from typing import ContextManager, Dict, Iterable, List, TypeVar

import numpy
from openff.qcsubmit.results import OptimizationResultCollection
from openff.toolkit import Molecule, Quantity
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from yammbs._db import (
    DBBase,
    DBMMConformerRecord,
    DBMoleculeRecord,
    DBQMConformerRecord,
)
from yammbs._session import DBSessionManager
from yammbs._types import Pathlike
from yammbs.analysis import (
    DDE,
    ICRMSD,
    RMSD,
    TFD,
    DDECollection,
    ICRMSDCollection,
    RMSDCollection,
    TFDCollection,
    get_internal_coordinate_rmsds,
    get_rmsd,
    get_tfd,
)
from yammbs.cached_result import CachedResultCollection
from yammbs.exceptions import DatabaseExistsError
from yammbs.inputs import QCArchiveDataset
from yammbs.models import MMConformerRecord, MoleculeRecord, QMConformerRecord

LOGGER = logging.getLogger(__name__)

MS = TypeVar("MS", bound="MoleculeStore")


class MoleculeStore:
    def __len__(self):
        with self._get_session() as db:
            return db.db.query(DBMoleculeRecord.mapped_smiles).count()

    def __init__(self, database_path: Pathlike = "molecule-store.sqlite"):
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
            self.db_version = db.check_version()
            self.general_provenance = db.get_general_provenance()
            self.software_provenance = db.get_software_provenance()

    @contextmanager
    def _get_session(self) -> ContextManager[Session]:
        session = self._sessionmaker()
        try:
            yield DBSessionManager(session)
            session.commit()
        except BaseException as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def _set_provenance(
        self,
        general_provenance: Dict[str, str],
        software_provenance: Dict[str, str],
    ):
        """Set the stores provenance information.

        Parameters
        ----------
        general_provenance
            A dictionary storing provenance about the store such as the author,
            when it was generated etc.
        software_provenance
            A dictionary storing the provenance of the software and packages used
            to generate the data in the store.
        """

        with self._get_session() as db:
            db.set_provenance(
                general_provenance=general_provenance,
                software_provenance=software_provenance,
            )
            self.general_provenance = db.get_general_provenance()
            self.software_provenance = db.get_software_provenance()

    def store(
        self,
        records: Iterable[MoleculeRecord],
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

    def store_qcarchive(
        self,
        records: Iterable[QMConformerRecord],
    ):
        if isinstance(records, QMConformerRecord):
            records = [records]

        with self._get_session() as db:
            for record in records:
                if db._qm_conformer_already_exists(record.qcarchive_id):
                    continue
                else:
                    db.store_qm_conformer_record(record)

    def store_conformer(
        self,
        records: Iterable[MMConformerRecord],
    ):
        if isinstance(records, MMConformerRecord):
            records = [records]

        with self._get_session() as db:
            for record in records:
                if db._mm_conformer_already_exists(
                    record.qcarchive_id,
                    record.force_field,
                ):
                    continue
                else:
                    db.store_mm_conformer_record(record)

    def get_molecule_ids(self) -> list[int]:
        """
        Get the molecule IDs of all records in the store.

        These are likely to be integers sequentially incrementing from 1, but that
        is not guaranteed.
        """
        with self._get_session() as db:
            return [molecule_id for (molecule_id,) in db.db.query(DBMoleculeRecord.id).distinct()]

    def get_smiles(self) -> List[str]:
        """Get the (mapped) smiles of all records in the store."""
        with self._get_session() as db:
            return [smiles for (smiles,) in db.db.query(DBMoleculeRecord.mapped_smiles).distinct()]

    def get_inchi_keys(self) -> List[str]:
        """Get the inchi keys of all records in the store."""
        with self._get_session() as db:
            return [inchi_key for (inchi_key,) in db.db.query(DBMoleculeRecord.inchi_key).distinct()]

    # TODO: Allow by multiple selectors (smiles: list[str])
    def get_molecule_id_by_smiles(self, smiles: str) -> int:
        with self._get_session() as db:
            return next(id for (id,) in db.db.query(DBMoleculeRecord.id).filter_by(mapped_smiles=smiles).all())

    # TODO: Allow by multiple selectors (id: list[int])
    def get_smiles_by_molecule_id(self, id: int) -> str:
        with self._get_session() as db:
            return next(smiles for (smiles,) in db.db.query(DBMoleculeRecord.mapped_smiles).filter_by(id=id).all())

    def get_molecule_id_by_inchi_key(self, inchi_key: str) -> int:
        with self._get_session() as db:
            return next(id for (id,) in db.db.query(DBMoleculeRecord.id).filter_by(inchi_key=inchi_key).all())

    def get_inchi_key_by_molecule_id(self, id: int) -> str:
        with self._get_session() as db:
            return next(inchi_key for (inchi_key,) in db.db.query(DBMoleculeRecord.inchi_key).filter_by(id=id).all())

    def get_qcarchive_ids_by_molecule_id(self, id: int) -> list[str]:
        with self._get_session() as db:
            return [
                qcarchive_id
                for (qcarchive_id,) in db.db.query(DBQMConformerRecord.qcarchive_id)
                .filter_by(parent_id=id)
                .order_by(DBQMConformerRecord.qcarchive_id)
                .all()
            ]

    # TODO: if this can take a list of ids, should it sort by QCArchive ID
    def get_molecule_id_by_qcarchive_id(self, id: str) -> int:
        with self._get_session() as db:
            return next(
                molecule_id
                for (molecule_id,) in db.db.query(DBQMConformerRecord.parent_id).filter_by(qcarchive_id=id).all()
            )

    def get_qm_conformers_by_molecule_id(self, id: int) -> list:
        with self._get_session() as db:
            return [
                conformer
                for (conformer,) in db.db.query(DBQMConformerRecord.coordinates)
                .filter_by(parent_id=id)
                .order_by(DBQMConformerRecord.qcarchive_id)
                .all()
            ]

    def get_force_fields(
        self,
    ) -> list[str]:
        """Return a list of all force fields with some conformers stored."""
        with self._get_session() as db:
            return [
                force_field
                for (force_field,) in db.db.query(
                    DBMMConformerRecord.force_field,
                ).distinct()
            ]

    def get_mm_conformers_by_molecule_id(
        self,
        id: int,
        force_field: str,
    ) -> list:
        with self._get_session() as db:
            return [
                conformer
                for (conformer,) in db.db.query(DBMMConformerRecord.coordinates)
                .filter_by(parent_id=id)
                .filter_by(force_field=force_field)
                .order_by(DBMMConformerRecord.qcarchive_id)
                .all()
            ]

    def get_qm_conformer_by_qcarchive_id(self, id: int):
        with self._get_session() as db:
            return next(
                conformer
                for (conformer,) in db.db.query(DBQMConformerRecord.coordinates).filter_by(qcarchive_id=id).all()
            )

    def get_mm_conformer_by_qcarchive_id(self, id: int, force_field: str):
        with self._get_session() as db:
            return next(
                conformer
                for (conformer,) in db.db.query(DBMMConformerRecord.coordinates)
                .filter_by(qcarchive_id=id)
                .filter_by(force_field=force_field)
                .all()
            )

    # TODO: Allow by multiple selectors (id: list[int])
    def get_qm_energies_by_molecule_id(self, id: int) -> list[float]:
        """Return a list of all QM energies for a molecule, stored as floats, implicitly in kcal/mol."""
        with self._get_session() as db:
            return [
                energy
                for (energy,) in db.db.query(DBQMConformerRecord.energy)
                .filter_by(parent_id=id)
                .order_by(DBQMConformerRecord.qcarchive_id)
                .all()
            ]

    # TODO: Allow by multiple selectors (id: list[int])
    def get_mm_energies_by_molecule_id(
        self,
        id: int,
        force_field: str,
    ) -> list[float]:
        """Return a list of all QM energies for a molecule, stored as floats, implicitly in kcal/mol."""
        with self._get_session() as db:
            return [
                energy
                for (energy,) in db.db.query(DBMMConformerRecord.energy)
                .filter_by(parent_id=id)
                .filter_by(force_field=force_field)
                .order_by(DBMMConformerRecord.qcarchive_id)
                .all()
            ]

    def get_qm_conformer_records_by_molecule_id(
        self,
        molecule_id: int,
    ) -> list[QMConformerRecord]:
        with self._get_session() as db:
            contents = [
                QMConformerRecord(
                    molecule_id=molecule_id,
                    qcarchive_id=x.qcarchive_id,
                    mapped_smiles=x.mapped_smiles,
                    coordinates=x.coordinates,
                    energy=x.energy,
                )
                for x in db.db.query(DBQMConformerRecord)
                .filter_by(parent_id=molecule_id)
                .order_by(DBQMConformerRecord.qcarchive_id)
                .all()
            ]

        return contents

    def get_mm_conformer_records_by_molecule_id(
        self,
        molecule_id: int,
        force_field: str,
    ) -> list[MMConformerRecord]:
        with self._get_session() as db:
            contents = [
                MMConformerRecord(
                    molecule_id=molecule_id,
                    qcarchive_id=x.qcarchive_id,
                    force_field=x.force_field,
                    mapped_smiles=x.mapped_smiles,
                    coordinates=x.coordinates,
                    energy=x.energy,
                )
                for x in db.db.query(DBMMConformerRecord)
                .filter_by(parent_id=molecule_id)
                .filter_by(force_field=force_field)
                .order_by(DBMMConformerRecord.qcarchive_id)
                .all()
            ]
        return contents

    @classmethod
    def from_qcsubmit_collection(
        cls,
        collection: OptimizationResultCollection,
        database_name: str,
    ) -> MS:
        from tqdm import tqdm

        if pathlib.Path(database_name).exists():
            raise DatabaseExistsError(f"Database {database_name} already exists.")

        store = cls(database_name)

        for qcarchive_record, molecule in tqdm(
            collection.to_records(),
            desc="Converting records to molecules",
        ):
            # _toolkit_registry_manager could go here

            molecule_record = MoleculeRecord.from_molecule(molecule)

            store.store(molecule_record)

            store.store_qcarchive(
                QMConformerRecord.from_qcarchive_record(
                    molecule_id=store.get_molecule_id_by_smiles(
                        molecule_record.mapped_smiles,
                    ),
                    mapped_smiles=molecule_record.mapped_smiles,
                    qc_record=qcarchive_record,
                    coordinates=molecule.conformers[0],
                ),
            )

        return store

    @classmethod
    def from_cached_result_collection(
        cls,
        collection: CachedResultCollection,
        database_name: str,
    ) -> MS:
        from tqdm import tqdm

        if pathlib.Path(database_name).exists():
            raise DatabaseExistsError(f"Database {database_name} already exists.")

        store = cls(database_name)

        # adapted from MoleculeRecord.from_molecule, MoleculeStore.store, and
        # DBSessionManager.store_molecule_record
        with store._get_session() as db:
            # instead of DBSessionManager._smiles_already_exists
            seen = set(db.db.query(DBMoleculeRecord.mapped_smiles))
            for rec in tqdm(collection.inner, desc="Storing molecules"):
                if rec.mapped_smiles in seen:
                    continue
                seen.add(rec.mapped_smiles)
                db_record = DBMoleculeRecord(
                    mapped_smiles=rec.mapped_smiles,
                    inchi_key=rec.inchi_key,
                )
                db.db.add(db_record)
                db.db.commit()

        # close the session here and re-open to make sure all of the molecule
        # IDs have been flushed to the db

        # adapted from MoleculeStore.store_qcarchive,
        # QMConformerRecord.from_qcarchive_record, and
        # DBSessionManager.store_qm_conformer_record
        with store._get_session() as db:
            seen = set(
                db.db.query(
                    DBQMConformerRecord.qcarchive_id,
                ),
            )
            # reversed so the first record encountered wins out. this matches
            # the behavior of the version that queries the db each time
            smiles_to_id = {
                smi: id
                for id, smi in reversed(
                    db.db.query(
                        DBMoleculeRecord.id,
                        DBMoleculeRecord.mapped_smiles,
                    ).all(),
                )
            }
            for record in tqdm(collection.inner, desc="Storing Records"):
                if record.qc_record_id in seen:
                    continue
                seen.add(record.qc_record_id)
                mol_id = smiles_to_id[record.mapped_smiles]
                db.db.add(
                    DBQMConformerRecord(
                        parent_id=mol_id,
                        qcarchive_id=record.qc_record_id,
                        mapped_smiles=record.mapped_smiles,
                        coordinates=record.coordinates,
                        energy=record.qc_record_final_energy,
                    ),
                )

        return store

    @classmethod
    def from_qcarchive_dataset(
        cls,
        dataset: QCArchiveDataset,
        database_name: str,
    ) -> MS:
        """
        Create a new MoleculeStore databset from YAMMBS's QCArchiveDataset model.

        Largely adopted from `from_qcsubmit_collection`.
        """
        from tqdm import tqdm

        if pathlib.Path(database_name).exists():
            raise DatabaseExistsError(f"Database {database_name} already exists.")

        store = cls(database_name)

        for qm_molecule in tqdm(dataset.qm_molecules, desc="Storing molecules"):
            molecule = Molecule.from_mapped_smiles(qm_molecule.mapped_smiles)
            molecule.add_conformer(Quantity(qm_molecule.coordinates, "angstrom"))

            molecule_record = MoleculeRecord.from_molecule(molecule)
            store.store(molecule_record)

            store.store_qcarchive(
                QMConformerRecord(
                    molecule_id=store.get_molecule_id_by_smiles(
                        molecule_record.mapped_smiles,
                    ),
                    qcarchive_id=qm_molecule.qcarchive_id,
                    mapped_smiles=qm_molecule.mapped_smiles,
                    coordinates=qm_molecule.coordinates,
                    energy=qm_molecule.final_energy,
                ),
            )

        return store

    def _map_inchi_keys_to_qm_conformers(self, force_field: str) -> dict[str, list]:
        inchi_keys = self.get_inchi_keys()

        mapping = defaultdict(list)

        for inchi_key in inchi_keys:
            molecule_id = self.get_molecule_id_by_inchi_key(inchi_key)

            # TODO: Should the session be inside or outside of the inchi loop?
            with self._get_session() as db:
                qm_conformers = [
                    {
                        "qcarchive_id": record.qcarchive_id,
                        "mapped_smiles": record.mapped_smiles,
                        "coordinates": record.coordinates,
                    }
                    for record in db.db.query(
                        DBQMConformerRecord,
                    )
                    .filter_by(parent_id=molecule_id)
                    .all()
                ]

                for qm_conformer in qm_conformers:
                    if not db._mm_conformer_already_exists(
                        qcarchive_id=qm_conformer["qcarchive_id"],
                        force_field=force_field,
                    ):
                        mapping[inchi_key].append(qm_conformer)
                    else:
                        pass

        return mapping

    def optimize_mm(
        self,
        force_field: str,
        n_processes: int = 2,
        chunksize=32,
    ):
        from yammbs._minimize import _minimize_blob

        inchi_key_qm_conformer_mapping = self._map_inchi_keys_to_qm_conformers(
            force_field=force_field,
        )

        if len(inchi_key_qm_conformer_mapping) == 0:
            return

        _minimized_blob = _minimize_blob(
            input=inchi_key_qm_conformer_mapping,
            force_field=force_field,
            n_processes=n_processes,
            chunksize=chunksize,
        )

        with self._get_session() as db:
            inchi_to_id: dict[str, int] = {
                inchi_key: id
                for (id, inchi_key) in reversed(
                    db.db.query(
                        DBMoleculeRecord.id,
                        DBMoleculeRecord.inchi_key,
                    ).all(),
                )
            }

        with self._get_session() as db:
            # from _mm_conformer_already_exists
            seen = set(
                db.db.query(
                    DBMMConformerRecord.qcarchive_id,
                ).filter_by(
                    force_field=force_field,
                ),
            )
            for result in _minimized_blob:
                if result.qcarchive_id in seen:
                    continue

                molecule_id = inchi_to_id[result.inchi_key]

                record = MMConformerRecord(
                    molecule_id=molecule_id,
                    qcarchive_id=result.qcarchive_id,
                    force_field=result.force_field,
                    mapped_smiles=result.mapped_smiles,
                    energy=result.energy,
                    coordinates=result.coordinates,
                )

                # inlined from MoleculeStore.store_conformer
                seen.add(record.qcarchive_id)

                db.store_mm_conformer_record(record)

    def get_dde(
        self,
        force_field: str,
        skip_check: bool = False,
    ) -> DDECollection:
        if not skip_check:
            self.optimize_mm(force_field=force_field)

        ddes = DDECollection()

        for inchi_key in self.get_inchi_keys():
            molecule_id = self.get_molecule_id_by_inchi_key(inchi_key)

            qcarchive_ids = self.get_qcarchive_ids_by_molecule_id(molecule_id)

            if len(qcarchive_ids) == 1:
                # There's only one conformer for this molecule
                # TODO: Quicker way of short-circuiting here
                continue

            # these functions should each return list[float],
            # implicitly both in kcal/mol
            qm_energies = numpy.array(
                self.get_qm_energies_by_molecule_id(
                    molecule_id,
                ),
            )

            mm_energies = numpy.array(
                self.get_mm_energies_by_molecule_id(
                    molecule_id,
                    force_field,
                ),
            )

            if len(mm_energies) != len(qm_energies):
                continue

            qm_minimum_index = qm_energies.argmin()

            mm_energies -= mm_energies[qm_minimum_index]
            qm_energies -= qm_energies[qm_minimum_index]

            mm_energies[qm_minimum_index] = numpy.nan
            qm_energies[qm_minimum_index] = numpy.nan

            for qm, mm, id in zip(
                qm_energies,
                mm_energies,
                qcarchive_ids,
            ):
                ddes.append(
                    DDE(
                        qcarchive_id=id,
                        difference=mm - qm,
                        force_field=force_field,
                    ),
                )

        return ddes

    def get_rmsd(
        self,
        force_field: str,
        skip_check: bool = False,
    ) -> RMSDCollection:
        if not skip_check:
            self.optimize_mm(force_field=force_field)

        rmsds = RMSDCollection()

        for inchi_key in self.get_inchi_keys():
            molecule = Molecule.from_inchi(inchi_key, allow_undefined_stereo=True)
            molecule_id = self.get_molecule_id_by_inchi_key(inchi_key)

            qcarchive_ids = self.get_qcarchive_ids_by_molecule_id(molecule_id)

            qm_conformers = self.get_qm_conformers_by_molecule_id(molecule_id)
            mm_conformers = self.get_mm_conformers_by_molecule_id(
                molecule_id,
                force_field,
            )

            for qm, mm, id in zip(
                qm_conformers,
                mm_conformers,
                qcarchive_ids,
            ):
                rmsds.append(
                    RMSD(
                        qcarchive_id=id,
                        rmsd=get_rmsd(molecule, qm, mm),
                        force_field=force_field,
                    ),
                )

        return rmsds

    def get_internal_coordinate_rmsd(
        self,
        force_field: str,
        skip_check: bool = False,
    ) -> ICRMSDCollection:
        if not skip_check:
            self.optimize_mm(force_field=force_field)

        icrmsds = ICRMSDCollection()

        for inchi_key in self.get_inchi_keys():
            molecule = Molecule.from_inchi(inchi_key, allow_undefined_stereo=True)
            molecule_id = self.get_molecule_id_by_inchi_key(inchi_key)

            qcarchive_ids = self.get_qcarchive_ids_by_molecule_id(molecule_id)

            qm_conformers = self.get_qm_conformers_by_molecule_id(molecule_id)
            mm_conformers = self.get_mm_conformers_by_molecule_id(
                molecule_id,
                force_field,
            )

            for qm, mm, id in zip(
                qm_conformers,
                mm_conformers,
                qcarchive_ids,
            ):
                icrmsds.append(
                    ICRMSD(
                        qcarchive_id=id,
                        icrmsd=get_internal_coordinate_rmsds(molecule, qm, mm),
                        force_field=force_field,
                    ),
                )

        return icrmsds

    def get_tfd(
        self,
        force_field: str,
        skip_check: bool = False,
    ) -> TFDCollection:
        if not skip_check:
            self.optimize_mm(force_field=force_field)

        tfds = TFDCollection()

        for inchi_key in self.get_inchi_keys():
            molecule = Molecule.from_inchi(inchi_key, allow_undefined_stereo=True)
            molecule_id = self.get_molecule_id_by_inchi_key(inchi_key)

            qcarchive_ids = self.get_qcarchive_ids_by_molecule_id(molecule_id)

            qm_conformers = self.get_qm_conformers_by_molecule_id(molecule_id)
            mm_conformers = self.get_mm_conformers_by_molecule_id(
                molecule_id,
                force_field,
            )

            for qm, mm, id in zip(
                qm_conformers,
                mm_conformers,
                qcarchive_ids,
            ):
                try:
                    tfds.append(
                        TFD(
                            qcarchive_id=id,
                            tfd=get_tfd(molecule, qm, mm),
                            force_field=force_field,
                        ),
                    )
                except Exception as e:
                    logging.warning(f"Molecule {inchi_key} failed with {e!s}")

        return tfds


def smiles_to_inchi_key(smiles: str) -> str:
    from openff.toolkit import Molecule

    return Molecule.from_smiles(smiles, allow_undefined_stereo=True).to_inchi(
        fixed_hydrogens=True,
    )
