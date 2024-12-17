"""A module for managing the database session."""

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, NamedTuple

from yammbs._db import (
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

    from yammbs.models import MMConformerRecord, MoleculeRecord, QMConformerRecord


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
            },
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
            f"be added in future versions.",
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
        return {provenance.key: provenance.value for provenance in self.db_info.general_provenance}

    def get_software_provenance(self):
        return {provenance.key: provenance.value for provenance in self.db_info.software_provenance}

    def set_provenance(
        self,
        general_provenance: Dict[str, str],
        software_provenance: Dict[str, str | None],
    ):
        self.db_info.general_provenance = [
            DBGeneralProvenance(key=key, value=value) for key, value in general_provenance.items()
        ]
        self.db_info.software_provenance = [
            DBSoftwareProvenance(key=key, value=value) for key, value in software_provenance.items()
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

    def store_qm_conformer_record(
        self,
        record: "QMConformerRecord",
    ):
        self.db.add(
            DBQMConformerRecord(
                parent_id=record.molecule_id,
                qcarchive_id=record.qcarchive_id,
                mapped_smiles=record.mapped_smiles,
                coordinates=record.coordinates,
                energy=record.energy,
            ),
        )

    def _qm_conformer_already_exists(
        self,
        qcarchive_id: int,
    ) -> bool:
        records = self.db.query(
            DBQMConformerRecord.qcarchive_id,
        ).filter_by(
            qcarchive_id=qcarchive_id,
        )

        return records.count() > 0

    def store_mm_conformer_record(
        self,
        record: "MMConformerRecord",
    ):
        self.db.add(
            DBMMConformerRecord(
                parent_id=record.molecule_id,
                qcarchive_id=record.qcarchive_id,
                force_field=record.force_field,
                mapped_smiles=record.mapped_smiles,
                coordinates=record.coordinates,
                energy=record.energy,
            ),
        )

    def _mm_conformer_already_exists(
        self,
        qcarchive_id: int,
        force_field: str,
    ) -> bool:
        records = (
            self.db.query(
                DBMMConformerRecord.qcarchive_id,
            )
            .filter_by(
                qcarchive_id=qcarchive_id,
            )
            .filter_by(
                force_field=force_field,
            )
        )

        return records.count() > 0
