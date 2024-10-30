from sqlalchemy import Column, Float, ForeignKey, Integer, PickleType, String
from sqlalchemy.orm import declarative_base  # type: ignore[attr-defined]

from yammbs.torsion.models import MMTorsionPointRecord, QMTorsionPointRecord

DBBase = declarative_base()

DB_VERSION = 1


class DBTorsionProfileRecord(DBBase):
    __tablename__ = "torsion_points"

    # is this like a molecule ID or like a QCArchive ID?
    parent_id: int = Column(Integer, ForeignKey("molecules.id"), nullable=False, index=True)

    # TODO: Store QCArchive ID


class DBQMTorsionPointRecord(DBBase):  # type: ignore
    __tablename__ = "qm_points"

    # is this like a molecule ID or like a QCArchive ID?
    parent_id = Column(Integer, ForeignKey("molecules.id"), nullable=False, index=True)

    # TODO: This should be a tuple of floats for 2D grids?
    grid_id = Column(Float, primary_key=True, index=True)

    coordinates = Column(PickleType, nullable=False)


class DBMMTorsionPointRecord(DBBase):  # type: ignore
    __tablename__ = "mm_points"

    # is this like a molecule ID or like a QCArchive ID?
    parent_id = Column(Integer, ForeignKey("molecules.id"), nullable=False, index=True)

    # TODO: This should be a tuple of floats for 2D grids?
    grid_id = Column(Float, primary_key=True, index=True)

    # TODO: Any more information in the "MM" optimization to store?
    force_field = Column(String, nullable=False)

    coordinates = Column(PickleType, nullable=False)


class DBMoleculeRecord(DBBase):  # type: ignore
    __tablename__ = "molecules"

    id = Column(Integer, primary_key=True, index=True)

    inchi_key = Column(String, nullable=False, index=True)
    mapped_smiles = Column(String, nullable=False)

    def store_qm_point_records(self, records: list[QMTorsionPointRecord]):
        if not isinstance(records, list):
            raise ValueError("records must be a list")

        for record in records:
            db_record = DBQMTorsionPointRecord()

            self.qm_points.append(db_record)

    def store_mm_point_records(self, records: list[MMTorsionPointRecord]):
        if not isinstance(records, list):
            raise ValueError("records must be a list")

        for record in records:
            db_record = DBMMTorsionPointRecord()

            self.mm_points.append(db_record)
