from sqlalchemy import Column, Float, ForeignKey, Integer, PickleType, String
from sqlalchemy.orm import declarative_base  # type: ignore[attr-defined]

DBBase = declarative_base()

DB_VERSION = 1


class DBTorsionRecord(DBBase):  # type: ignore
    __tablename__ = "torsion_molecules"

    id = Column(Integer, primary_key=True, index=True)

    inchi_key = Column(String, nullable=False, index=True)
    mapped_smiles = Column(String, nullable=False)

    dihedral_indices = Column(PickleType, nullable=False)


class DBTorsionProfileRecord(DBBase):  # type: ignore
    __tablename__ = "torsion_points"

    id = Column(Integer, primary_key=True, index=True)

    # is this like a molecule ID or like a QCArchive ID?
    parent_id = Column(Integer, ForeignKey("molecules.id"), nullable=False, index=True)

    # TODO: Store QCArchive ID


class DBQMTorsionPointRecord(DBBase):  # type: ignore
    __tablename__ = "qm_points"

    id = Column(Integer, primary_key=True, index=True)

    # is this like a molecule ID or like a QCArchive ID?
    parent_id = Column(Integer, ForeignKey("molecules.id"), nullable=False, index=True)

    # TODO: This should be a tuple of floats for 2D grids?
    grid_id = Column(Float, nullable=False)

    coordinates = Column(PickleType, nullable=False)

    energy = Column(Float, nullable=False)


class DBMMTorsionPointRecord(DBBase):  # type: ignore
    __tablename__ = "mm_points"

    id = Column(Integer, primary_key=True, index=True)

    # is this like a molecule ID or like a QCArchive ID?
    parent_id = Column(Integer, ForeignKey("molecules.id"), nullable=False, index=True)

    # TODO: This should be a tuple of floats for 2D grids?
    grid_id = Column(Float, nullable=False)

    # TODO: Any more information in the "MM" optimization to store?
    force_field = Column(String, nullable=False)

    coordinates = Column(PickleType, nullable=False)

    energy = Column(Float, nullable=False)


class DBMoleculeRecord(DBBase):  # type: ignore
    __tablename__ = "molecules"

    id = Column(Integer, primary_key=True, index=True)

    inchi_key = Column(String, nullable=False, index=True)
    mapped_smiles = Column(String, nullable=False)
