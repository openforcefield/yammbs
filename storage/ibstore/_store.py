import logging
import pathlib
from collections import defaultdict
from contextlib import contextmanager
from typing import ContextManager, Dict, List, Optional, Tuple

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from ibstore._db import DBBase, DBMoleculeRecord, DBPartialChargeSet
from ibstore._session import DBSessionManager
from ibstore._types import Pathlike
from ibstore.record import MoleculeRecord

LOGGER = logging.getLogger(__name__)


class MoleculeStore:
    """A class used to store and retrieve calculated sets of partial charges and
    Wiberg bond orders for sets of molecules.
    """

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

    def set_provenance(
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

    def get_charge_methods(self) -> List[str]:
        """A list of the methods used to compute the partial charges in the store."""

        with self._get_session() as db:
            return [
                method
                for (method,) in db.db.query(DBPartialChargeSet.method).distinct()
            ]

    def get_smiles(self) -> List[str]:
        with self._get_session() as db:
            return [
                smiles
                for (smiles,) in db.db.query(DBMoleculeRecord.mapped_smiles).distinct()
            ]

    def retrieve(
        self,
        qcarchive_id: Optional[str] = None,
    ) -> List[MoleculeRecord]:
        """Retrieve records stored in this data store

        Args:
            partial_charge_methods: The (optional) list of charge methods to retrieve
                from the store. By default (`None`) all charges will be returned.
            bond_order_methods: The (optional) list of bond order methods to retrieve
                from the store. By default (`None`) all bond orders will be returned.

        Returns:
            The retrieved records.
        """
        raise NotImplementedError()

    def store(
        self,
        records: Tuple[str, MoleculeRecord] = tuple(),
        suppress_toolkit_warnings: bool = True,
    ):
        """Store the molecules and their computed properties in the data store.

        Parameters
        ----------
        records
            The QCArchive id and record of each molecule to store.
        suppress_toolkit_warnings: bool
            Whether to suppress toolkit warnings when loading molecules.
        """
        from openff.nagl.toolkits.openff import (
            capture_toolkit_warnings,
            smiles_to_inchi_key,
        )

        if isinstance(records, MoleculeRecord):
            records = [records]

        with capture_toolkit_warnings(run=suppress_toolkit_warnings):
            records_by_inchi_key = defaultdict(list)

            for record in tqdm(records, desc="grouping records to store by InChI key"):
                inchi_key = smiles_to_inchi_key(record.mapped_smiles)
                records_by_inchi_key[inchi_key].append(record)

            with self._get_session() as db:
                for inchi_key, inchi_records in tqdm(
                    records_by_inchi_key.items(), desc="storing grouped records"
                ):
                    db.store_records_with_inchi_key(inchi_key, inchi_records)
