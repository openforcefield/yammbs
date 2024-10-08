import logging
from typing import Dict, List

from sqlalchemy import Column, Float, ForeignKey, Integer, PickleType, String
from sqlalchemy.orm import declarative_base, relationship

from yammbs.models import MMConformerRecord, QMConformerRecord

DBBase = declarative_base()

DB_VERSION = 1

LOGGER = logging.getLogger(__name__)


class DBQMConformerRecord(DBBase):
    __tablename__ = "qm_conformers"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("molecules.id"), nullable=False, index=True)

    qcarchive_id = Column(Integer, nullable=False)

    mapped_smiles = Column(String, nullable=False)
    coordinates = Column(PickleType, nullable=False)
    energy = Column(Float, nullable=False)


class DBMMConformerRecord(DBBase):
    __tablename__ = "mm_conformers"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("molecules.id"), nullable=False, index=True)

    qcarchive_id = Column(Integer, nullable=False)
    force_field = Column(String, nullable=False)

    mapped_smiles = Column(String, nullable=False)
    coordinates = Column(PickleType, nullable=False)
    energy = Column(Float, nullable=False)


class DBMoleculeRecord(DBBase):
    __tablename__ = "molecules"

    id = Column(Integer, primary_key=True, index=True)

    inchi_key = Column(String, nullable=False, index=True)
    mapped_smiles = Column(String, nullable=False)

    def store_qm_conformer_records(self, records: List[QMConformerRecord]):
        if not isinstance(records, list):
            raise ValueError("records must be a list")
        # TODO: match conformers?
        for record in records:
            db_record = DBQMConformerRecord(
                qcarchive_id=record.qcarchive_id,
                coordinates=record.coordinates,
                energy=record.energy,
            )
            self.qm_conformers.append(db_record)

    def store_mm_conformer_records(self, records: List[MMConformerRecord]):
        if not isinstance(records, list):
            raise ValueError("records must be a list")
        # TODO: match conformers?
        for record in records:
            db_record = DBMMConformerRecord(
                qcarchive_id=record.qcarchive_id,
                force_field=record.force_field,
                coordinates=record.coordinates,
                energy=record.energy,
            )
            self.mm_conformers.append(db_record)


class DBGeneralProvenance(DBBase):
    __tablename__ = "general_provenance"

    key = Column(String, primary_key=True, index=True, unique=True)
    value = Column(String, nullable=False)

    parent_id = Column(Integer, ForeignKey("db_info.version"))


class DBSoftwareProvenance(DBBase):
    __tablename__ = "software_provenance"

    key = Column(String, primary_key=True, index=True, unique=True)
    value = Column(String, nullable=True)

    parent_id = Column(Integer, ForeignKey("db_info.version"))


class DBInformation(DBBase):
    """A class which keeps track of the current database
    settings.
    """

    __tablename__ = "db_info"

    version = Column(Integer, primary_key=True)

    general_provenance = relationship(
        "DBGeneralProvenance",
        cascade="all, delete-orphan",
    )
    software_provenance = relationship(
        "DBSoftwareProvenance",
        cascade="all, delete-orphan",
    )


def _match_conformers(
    indexed_mapped_smiles: str,
    db_conformers: List,
    query_conformers: List,
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
        indexed_mapped_smiles,
        allow_undefined_stereo=True,
    )

    # See if any of the conformers to add are already in the DB.
    matches = {}

    for q_index, query in enumerate(query_conformers):
        for db_index, db_conformer in enumerate(db_conformers):
            if is_conformer_identical(
                molecule,
                query.coordinates,
                db_conformer.coordinates,
            ):
                matches[q_index] = db_index
                break
    return matches
