import logging
import pathlib
from contextlib import contextmanager
from typing import ContextManager, Dict, Iterable, List, TypeVar

from openff.qcsubmit.results import OptimizationResultCollection
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ibstore._db import DBBase, DBMoleculeRecord
from ibstore._session import DBSessionManager
from ibstore._types import Pathlike
from ibstore.models import MMConformerRecord, MoleculeRecord, QMConformerRecord

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
                "Only paths to SQLite databases ending in .sqlite "
                f"are supported. Given: {database_path}"
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

        raise NotImplementedError()

    def store_minimized_conformer(
        self,
        records: Iterable[MMConformerRecord],
    ):
        if isinstance(records, MMConformerRecord):
            records = [records]

        raise NotImplementedError()

    def get_smiles(self) -> List[str]:
        """Get the (mapped) smiles of all records in the store."""
        with self._get_session() as db:
            return [
                smiles
                for (smiles,) in db.db.query(DBMoleculeRecord.mapped_smiles).distinct()
            ]

    def get_molecule_id_by_smiles(self, smiles: str) -> str:
        with self._get_session() as db:
            return [
                id
                for (id,) in db.db.query(DBMoleculeRecord.id)
                .filter_by(mapped_smiles=smiles)
                .all()
            ][0]

    @classmethod
    def from_qcsubmit_collection(
        cls,
        collection: OptimizationResultCollection,
        database_name: str,
    ) -> MS:
        store = cls("test.sqlite")

        for qcarchive_record, molecule in collection.to_records():
            # _toolkit_registry_manager could go here

            molecule_record = MoleculeRecord.from_molecule(molecule)

            store.store(molecule_record)

            store.store_qcarchive(
                QMConformerRecord.from_qcarchive_record(qcarchive_record),
            )

        return store


def smiles_to_inchi_key(smiles: str) -> str:
    from openff.toolkit import Molecule

    return Molecule.from_smiles(smiles, allow_undefined_stereo=True).to_inchikey(
        fixed_hydrogens=True
    )
