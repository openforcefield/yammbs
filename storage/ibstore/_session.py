"""A module for managing the database session."""
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional

from ibstore._db import (
    DB_VERSION,
    DBGeneralProvenance,
    DBInformation,
    DBMMConformerRecord,
    DBMoleculeRecord,
    DBQMConformerRecord,
    DBSoftwareProvenance,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from ibstore.models import MMConformerRecord, MoleculeRecord, QMConformerRecord


class DBQueryResult(NamedTuple):
    """A named tuple representing a single row of a database query."""

    molecule_id: int
    molecule_smiles: str
    molecule_inchi: str

    def to_nested_dict(self) -> dict[int, dict[str, str]]:
        return {
            self.molecule_id: {
                "mapped_smiles": self.molecule_smiles,
                "inchi": self.molecule_inchi,
            }
        }


class IncompatibleDBVersion(ValueError):
    """An exception raised when attempting to load a store whose
    version does not match the version expected by the framework.
    """

    def __init__(self, found_version: int, expected_version: int):
        """

        Parameters
        ----------
        found_version
            The version of the database being loaded.
        expected_version
            The expected version of the database.
        """

        super().__init__(
            f"The database being loaded is currently at version {found_version} "
            f"while the framework expects a version of {expected_version}. There "
            f"is no way to upgrade the database at this time, although this may "
            f"be added in future versions."
        )

        self.found_version = found_version
        self.expected_version = expected_version


class DBSessionManager:
    @staticmethod
    def map_records_by_smiles(
        db_records: List[DBMoleculeRecord],
    ) -> Dict[str, List[DBMoleculeRecord]]:
        """Maps a list of DB records by their SMILES representation.

        Parameters
        ----------
        records
            The records to map.

        Returns
        -------
        A dictionary mapping the SMILES representation of a record to the record.
        """

        from openff.toolkit.topology import Molecule

        records = defaultdict(list)
        for record in db_records:
            offmol = Molecule.from_smiles(
                record.mapped_smiles,
                allow_undefined_stereo=True,
            )
            canonical_smiles = offmol.to_smiles(mapped=False)
            records[canonical_smiles].append(record)
        return records

    def __init__(self, session: "Session"):
        self.session = session
        self._db_info = None

    def check_version(self, version=DB_VERSION):
        """Checks that the database is at the expected version."""
        if not self.db_info:
            db_info = DBInformation(version=version)
            self.db.add(db_info)
            self._db_info = db_info

        if self.db_info.version != version:
            raise IncompatibleDBVersion(self.db_info.version, version)
        return self.db_info.version

    def get_general_provenance(self):
        return {
            provenance.key: provenance.value
            for provenance in self.db_info.general_provenance
        }

    def get_software_provenance(self):
        return {
            provenance.key: provenance.value
            for provenance in self.db_info.software_provenance
        }

    def set_provenance(
        self,
        general_provenance: Dict[str, str],
        software_provenance: Dict[str, str],
    ):
        self.db_info.general_provenance = [
            DBGeneralProvenance(key=key, value=value)
            for key, value in general_provenance.items()
        ]
        self.db_info.software_provenance = [
            DBSoftwareProvenance(key=key, value=value)
            for key, value in software_provenance.items()
        ]

    @property
    def db_info(self):
        if self._db_info is None:
            self._db_info = self.db.query(DBInformation).first()
        return self._db_info

    @property
    def db(self):
        return self.session

    def _smiles_already_exists(
        self,
        smiles: str,
    ) -> bool:
        records = self.db.query(
            DBMoleculeRecord.mapped_smiles,
        ).filter_by(
            mapped_smiles=smiles,
        )

        return records.count() > 0

    def store_molecule_record(
        self,
        record: "MoleculeRecord",
    ):
        if self._smiles_already_exists(smiles=record.mapped_smiles):
            # TODO: log this
            return

        db_record = DBMoleculeRecord(
            mapped_smiles=record.mapped_smiles,
            inchi_key=record.inchi_key,
        )

        # db_record.store_qm_conformer_records([record.conformer])
        # db_record.store_mm_conformer_records([record.minimized_conformer])

        self.db.add(db_record)

    def store_qm_conformer_records(
        self,
        record: "QMConformerRecord",
        molecule_id: int,
    ):
        self.db.add(
            DBQMConformerRecord(
                parent_id=molecule_id,
                qcarchive_id=record.qcarchive_id,
                coordinates=record.coordinates,
                energy=record.energy,
            )
        )

    def store_mm_conformer_records(
        self,
        record: "MMConformerRecord",
        molecule_id: int,
    ):
        self.db.add(
            DBMMConformerRecord(
                parent_id=molecule_id,
                coordinates=record.coordinates,
                energy=record.energy,
            )
        )

    def store_records_with_smiles(
        self,
        inchi_key: str,
        records: List["MoleculeRecord"],
        existing_db_record: Optional[DBMoleculeRecord] = None,
    ):
        """Stores a set of records which all store information for molecules with the
        same SMILES representation AND the same fixed hydrogen InChI key.

        Parameters
        ----------
        inchi_key: str
            The **fixed hydrogen** InChI key representation of the molecule stored in
            the records.
        records: List[MoleculeRecord]
            The records to store.
        existing_db_record: Optional[DBMoleculeRecord]
            An optional existing DB record to check
        """

        if existing_db_record is None:
            existing_db_record = DBMoleculeRecord(
                # inchi_key=inchi_key,
                mapped_smiles=records[0].mapped_smiles,
                qcarchive_id=records[0].qcarchive_id,
                qcarchive_energy=records[0].qcarchive_energy,
            )

        # Retrieve the DB indexed SMILES that defines the ordering the atoms in each
        # record should have and re-order the incoming records to match.
        expected_smiles = existing_db_record.mapped_smiles

        conformer_records = [
            conformer_record
            for record in records
            for conformer_record in record.reorder(expected_smiles).conformers
        ]

        existing_db_record.store_conformer_records(conformer_records)
        self.db.add(existing_db_record)

    def store_records_with_inchi_key(
        self,
        inchi_key: str,
        records: List["MoleculeRecord"],
    ):
        """Stores a set of records which all store information for molecules with the
        same fixed hydrogen InChI key.

        Parameters
        ----------
        inchi_key: str
            The **fixed hydrogen** InChI key representation of the molecule stored in
            the records.
        records: List[MoleculeRecord]
            The records to store.
        """

        existing_db_records: List[DBMoleculeRecord] = (
            self.db.query(DBMoleculeRecord)
            .filter(DBMoleculeRecord.inchi_key == inchi_key)
            .all()
        )

        db_records_by_smiles = self.map_records_by_smiles(existing_db_records)
        # Sanity check that no two DB records have the same InChI key AND the
        # same canonical SMILES pattern.
        multiple = [
            smiles
            for smiles, dbrecords in db_records_by_smiles.items()
            if len(dbrecords) > 1
        ]
        if multiple:
            raise RuntimeError(
                "The database is not self consistent."
                "There are multiple records with the same InChI key and SMILES."
                f"InChI key: {inchi_key} and SMILES: {multiple}"
            )
        db_records_by_smiles = {k: v[0] for k, v in db_records_by_smiles.items()}

        records_by_smiles = self.map_records_by_smiles(records)
        for smiles, smiles_records in records_by_smiles.items():
            db_record = db_records_by_smiles.get(smiles, None)
            self.store_records_with_smiles(inchi_key, smiles_records, db_record)
