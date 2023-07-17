import logging
from typing import Dict, List

from ibstore.record import ConformerRecord, MinimizedConformerRecord
from sqlalchemy import Column, ForeignKey, Integer, PickleType, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

DBBase = declarative_base()

DB_VERSION = 1

LOGGER = logging.getLogger(__name__)


class DBConformerRecord(DBBase):
    __tablename__ = "conformers"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("molecules.id"), nullable=False, index=True)

    coordinates = Column(PickleType, nullable=False)


class DBMinimizedConformerRecord(DBBase):
    __tablename__ = "minimized_conformers"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("molecules.id"), nullable=False, index=True)

    coordinates = Column(PickleType, nullable=False)


class DBMoleculeRecord(DBBase):
    __tablename__ = "molecules"

    id = Column(Integer, primary_key=True, index=True)

    qcarchive_id = Column(String(20), nullable=False)
    qcarchive_energy = Column(Float(24), nullable=False)

    minimized_energy = Column(Float(24))

    # inchi_key = Column(String(20), nullable=False, index=True)
    mapped_smiles = Column(String, nullable=False)

    conformer = relationship("DBConformerRecord", cascade="all, delete-orphan")
    minimized_conformer = relationship(
        "DBMinimizedConformerRecord", cascade="all, delete-orphan"
    )

    def store_conformer_records(self, records: List[ConformerRecord]):
        """Store a set of conformer records in an existing DB molecule record."""

        if len(self.conformers) > 0:
            LOGGER.warning(
                f"An entry for {self.mapped_smiles} is already present in the molecule store. "
                f"Trying to find matching conformers."
            )

        conformer_matches = match_conformers(
            self.mapped_smiles, self.conformers, records
        )

        # Create new database conformers for those unmatched conformers.
        for i, record in enumerate(records):
            if i in conformer_matches:
                continue

            db_conformer = DBConformerRecord(coordinates=record.coordinates)
            self.conformers.append(db_conformer)
            conformer_matches[i] = len(self.conformers) - 1

        DB_CLASSES = {}

        for index, db_index in conformer_matches.items():
            db_record: DBConformerRecord = self.conformers[db_index]
            record = records[index]
            for container_name, db_class in DB_CLASSES.items():
                db_record._store_new_data(
                    record, self.mapped_smiles, db_class, container_name
                )

    def store_minimized_conformer_records(
        self,
        records: List[MinimizedConformerRecord],
    ):
        raise NotImplementedError()


class DBGeneralProvenance(DBBase):
    __tablename__ = "general_provenance"

    key = Column(String, primary_key=True, index=True, unique=True)
    value = Column(String, nullable=False)

    parent_id = Column(Integer, ForeignKey("db_info.version"))


class DBSoftwareProvenance(DBBase):
    __tablename__ = "software_provenance"

    key = Column(String, primary_key=True, index=True, unique=True)
    value = Column(String, nullable=False)

    parent_id = Column(Integer, ForeignKey("db_info.version"))


class DBInformation(DBBase):
    """A class which keeps track of the current database
    settings.
    """

    __tablename__ = "db_info"

    version = Column(Integer, primary_key=True)

    general_provenance = relationship(
        "DBGeneralProvenance", cascade="all, delete-orphan"
    )
    software_provenance = relationship(
        "DBSoftwareProvenance", cascade="all, delete-orphan"
    )


def match_conformers(
    indexed_mapped_smiles: str,
    db_conformers: List[DBConformerRecord],
    query_conformers: List["ConformerRecord"],
) -> Dict[int, int]:
    """A method which attempts to match a set of new conformers to store with
    conformers already present in the database by comparing the RMS of the
    two sets.

    Args:
        indexed_mapped_smiles: The indexed mapped_smiles pattern associated with the conformers.
        db_conformers: The database conformers.
        conformers: The conformers to store.

    Returns:
        A dictionary which maps the index of a conformer to the index of a database
        conformer. The indices of conformers which do not match an existing database
        conformer are not included.
    """

    from openff.nagl.toolkits.openff import is_conformer_identical
    from openff.toolkit.topology import Molecule

    molecule = Molecule.from_mapped_smiles(
        indexed_mapped_smiles, allow_undefined_stereo=True
    )

    # See if any of the conformers to add are already in the DB.
    matches = {}

    for q_index, query in enumerate(query_conformers):
        for db_index, db_conformer in enumerate(db_conformers):
            if is_conformer_identical(
                molecule, query.coordinates, db_conformer.coordinates
            ):
                matches[q_index] = db_index
                break
    return matches
