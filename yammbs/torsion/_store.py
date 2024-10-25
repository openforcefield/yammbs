import logging
import pathlib
from contextlib import contextmanager
from typing import Generator, TypeVar

from openff.qcsubmit.results import TorsionDriveResultCollection
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from yammbs._db import (
    DBBase,
)
from yammbs._session import DBSessionManager
from yammbs._types import Pathlike
from yammbs.torsion.inputs import QCArchiveTorsionDataset

LOGGER = logging.getLogger(__name__)

MS = TypeVar("MS", bound="MoleculeStore")


class MoleculeStore:
    def __len__(self):
        with self._get_session() as _:
            raise NotImplementedError()

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
            self.db_version = db.check_version()
            self.general_provenance = db.get_general_provenance()
            self.software_provenance = db.get_software_provenance()

    @contextmanager
    def _get_session(self) -> Generator[DBSessionManager, None, None]:
        session = self._sessionmaker()
        try:
            yield DBSessionManager(session)
            session.commit()
        except BaseException as e:
            session.rollback()
            raise e
        finally:
            session.close()

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
