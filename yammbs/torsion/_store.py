import logging
import pathlib
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Literal, Self

import numpy
import pandas as pd
from numpy.typing import NDArray
from openff.qcsubmit.results import TorsionDriveResultCollection
from openff.toolkit import Molecule
from openff.toolkit.typing.engines.smirnoff.parameters import ProperTorsionHandler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from yammbs._forcefields import _lazy_load_force_field
from yammbs._molecule import _smiles_to_inchi_key
from yammbs._types import Pathlike
from yammbs.exceptions import DatabaseExistsError
from yammbs.torsion._db import (
    DBBase,
    DBMMTorsionPointRecord,
    DBQMTorsionPointRecord,
    DBTorsionRecord,
)
from yammbs.torsion._session import TorsionDBSessionManager
from yammbs.torsion.analysis import (
    RMSD,
    AnalysisMetricCollectionTypeVar,
    JSDistanceCollection,
    MeanErrorCollection,
    RMSDCollection,
    RMSECollection,
    _normalize,
)
from yammbs.torsion.inputs import QCArchiveTorsionDataset
from yammbs.torsion.models import (
    MMTorsionPointRecord,
    QMTorsionPointRecord,
    TorsionRecord,
)
from yammbs.torsion.outputs import Metric, MetricCollection, MinimizedTorsionDataset

LOGGER = logging.getLogger(__name__)


class TorsionStore:
    def __len__(self):
        with self._get_session() as db:
            return db.db.query(DBTorsionRecord.mapped_smiles).count()

    def __init__(self, database_path: Pathlike = "torsion-store.sqlite"):
        database_path = pathlib.Path(database_path)

        if not database_path.suffix.lower() == ".sqlite":
            raise NotImplementedError(
                f"Only paths to SQLite databases ending in .sqlite are supported. Given: {database_path}",
            )

        LOGGER.info(f"Creating a new TorsionStore at {database_path=}")

        self.database_url = f"sqlite:///{database_path.resolve()}"
        self.engine = create_engine(self.database_url)
        DBBase.metadata.create_all(self.engine)

        self._sessionmaker = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        with self._get_session() as db:
            assert db is not None
            # self.db_version = db.check_version()
            # self.general_provenance = db.get_general_provenance()
            # self.software_provenance = db.get_software_provenance()

    @contextmanager
    def _get_session(self) -> Generator[TorsionDBSessionManager, None, None]:
        session = self._sessionmaker()
        try:
            yield TorsionDBSessionManager(session)
            session.commit()
        except BaseException as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def store_torsion_record(
        self,
        records: TorsionRecord | Iterable[TorsionRecord],
    ):
        """Store molecules and their computed properties in the data store."""
        if isinstance(records, TorsionRecord):
            records = [records]

        with self._get_session() as db:
            for record in records:
                db.store_torsion_record(record)

    def store_qm_point(
        self,
        point: QMTorsionPointRecord,
    ):
        with self._get_session() as db:
            db.store_qm_torsion_point(point)

    def store_mm_point(
        self,
        point: MMTorsionPointRecord,
    ):
        with self._get_session() as db:
            db.store_mm_torsion_point(point)

    def get_torsion_ids(self) -> list[int]:
        """Get the molecule IDs of all records in the store.

        These are likely to be integers sequentially incrementing from 1, but that
        is not guaranteed.
        """
        # TODO: This isn't really a "molecule ID", it's more like a torsiondrive ID
        with self._get_session() as db:
            return [torsion_id for (torsion_id,) in db.db.query(DBTorsionRecord.torsion_id).distinct()]

    # TODO: Allow by multiple selectors (id: list[int])
    def get_smiles_by_torsion_id(self, torsion_id: int) -> str:
        with self._get_session() as db:
            return next(
                smiles
                for (smiles,) in db.db.query(DBTorsionRecord.mapped_smiles).filter_by(torsion_id=torsion_id).all()
            )

    def get_torsion_ids_by_smiles(self, smiles: str) -> list[int]:
        """Get all torsion IDs having a given mapped SMILES.

        Input mapped smiles must match an existing string in the database exactly.
        No chemical similarity check is performed.
        """
        with self._get_session() as db:
            return [
                torsion_id
                for (torsion_id,) in db.db.query(DBTorsionRecord.torsion_id).filter_by(mapped_smiles=smiles).all()
            ]

    # TODO: Allow by multiple selectors (id: list[int])
    def get_dihedral_indices_by_torsion_id(self, torsion_id: int) -> tuple[int, int, int, int]:
        with self._get_session() as db:
            return next(
                dihedral_indices
                for (dihedral_indices,) in db.db.query(DBTorsionRecord.dihedral_indices)
                .filter_by(torsion_id=torsion_id)
                .all()
            )

    def get_force_fields(
        self,
    ) -> list[str]:
        """Return a list of all force fields with some torsion points stored."""
        with self._get_session() as db:
            return [
                force_field
                for (force_field,) in db.db.query(
                    DBMMTorsionPointRecord.force_field,
                ).distinct()
            ]

    def get_qm_points_by_torsion_id(self, torsion_id: int) -> dict[float, NDArray]:
        with self._get_session() as db:
            return {
                grid_id: coordinates
                for (grid_id, coordinates) in db.db.query(
                    DBQMTorsionPointRecord.grid_id,
                    DBQMTorsionPointRecord.coordinates,
                )
                .filter_by(parent_id=torsion_id)
                .all()
            }

    def get_mm_points_by_torsion_id(
        self,
        torsion_id: int,
        force_field: str,
    ) -> dict[float, NDArray]:
        with self._get_session() as db:
            return {
                grid_id: coordinates
                for (grid_id, coordinates) in db.db.query(
                    DBMMTorsionPointRecord.grid_id,
                    DBMMTorsionPointRecord.coordinates,
                )
                .filter_by(parent_id=torsion_id)
                .filter_by(force_field=force_field)
                .all()
            }

    def get_qm_energies_by_torsion_id(self, torsion_id: int) -> dict[float, float]:
        with self._get_session() as db:
            return {
                grid_id: energy
                for (grid_id, energy) in db.db.query(
                    DBQMTorsionPointRecord.grid_id,
                    DBQMTorsionPointRecord.energy,
                )
                .filter_by(parent_id=torsion_id)
                .all()
            }

    def get_mm_energies_by_torsion_id(self, torsion_id: int, force_field: str) -> dict[float, float]:
        with self._get_session() as db:
            return {
                grid_id: energy
                for (grid_id, energy) in db.db.query(
                    DBMMTorsionPointRecord.grid_id,
                    DBMMTorsionPointRecord.energy,
                )
                .filter_by(parent_id=torsion_id)
                .filter_by(force_field=force_field)
                .all()
            }

    @classmethod
    def from_torsion_dataset(
        cls,
        dataset: QCArchiveTorsionDataset,
        database_name: str,
    ) -> Self:
        if pathlib.Path(database_name).exists():
            raise DatabaseExistsError(f"Database {database_name} already exists.")

        LOGGER.info(
            f"Creating a new TorsionStore at {database_name=} from a QCArchiveTorsionDataset "
            "(which is a YAMMBS model).",
        )

        store = cls(database_name)

        LOGGER.info("Iterating through qm_torsions field of QCArchiveTorsionDataset (which is a YAMMBS model).")

        for qm_torsion in dataset.qm_torsions:
            # TODO: Adapt this for non-QCArchive datasets like TorsionNet500
            this_id = qm_torsion.qcarchive_id

            torsion_record = TorsionRecord(
                torsion_id=this_id,
                mapped_smiles=qm_torsion.mapped_smiles,
                inchi_key=_smiles_to_inchi_key(qm_torsion.mapped_smiles),
                dihedral_indices=qm_torsion.dihedral_indices,
            )

            store.store_torsion_record(torsion_record)

            for angle in qm_torsion.coordinates:
                qm_point_record = QMTorsionPointRecord(
                    torsion_id=this_id,
                    grid_id=angle,  # TODO: This needs to be a tuple later for 2D scans
                    coordinates=qm_torsion.coordinates[angle],
                    energy=qm_torsion.energies[angle],
                )

                store.store_qm_point(qm_point_record)

        return store

    @classmethod
    def from_qcsubmit_collection(
        cls,
        collection: TorsionDriveResultCollection,
        database_name: str,
    ):
        """Convert a QCSubmit collection to TorsionDataset and use it to create a TorsionStore."""
        LOGGER.info(
            f"Creating a new TorsionStore at {database_name=} from a TorsionDriveResultCollection "
            "(which is a QCSubmit model).",
        )

        return cls.from_torsion_dataset(
            dataset=QCArchiveTorsionDataset.from_qcsubmit_collection(collection),
            database_name=database_name,
        )

    def optimize_mm(
        self,
        force_field: str,
        method: Literal["openmm", "geometric"] = "openmm",
        n_processes: int = 2,
        chunksize: int = 32,
        restraint_k: float = 0.0,
    ) -> None:
        """Run a constrained minimization of all torsion points.

        Parameters
        ----------
        force_field : str
            Force field to use for minimization.
        method : Literal["openmm", "geometric"]
            Minimization method to use. OpenMM constrains the positions of all
            atoms which define the torsion, while geomeTRIC only constrains the dihedral angle.
            The geomeTRIC approach is more rigorous but more expensive.
        n_processes : int
            Number of parallel processes.
        chunksize : int
            Chunk size for multiprocessing.
        restraint_k : float
            Restraint force constant in kcal/(mol*Angstrom^2) for atoms not
            in dihedral.

        Returns
        -------
        None

        """
        # TODO: Pass through more options for constrained minimization process?

        from yammbs.torsion._minimize import _minimize_torsions

        torsion_ids = self.get_torsion_ids()

        # TODO Do this by interacting with the database in one step?
        ids_to_minimize = [
            torsion_id
            for torsion_id in torsion_ids
            if len(self.get_mm_points_by_torsion_id(torsion_id, force_field)) == 0
        ]

        id_to_smiles = {torsion_id: self.get_smiles_by_torsion_id(torsion_id) for torsion_id in ids_to_minimize}
        id_to_dihedral_indices = {
            torsion_id: self.get_dihedral_indices_by_torsion_id(torsion_id) for torsion_id in ids_to_minimize
        }

        LOGGER.info(f"Setting up generator of data for minimization with {force_field=}")

        with self._get_session() as db:
            # TODO: Implement "seen" behavior to short-circuit already-optimized torsions
            data: Generator[
                tuple[
                    int,
                    str,
                    tuple[int, int, int, int],
                    float,
                    NDArray,
                    float,
                ],
                None,
                None,
            ] = (  # Probably a better way to do this with some proper database query with join
                (
                    torsion_id,
                    id_to_smiles[torsion_id],
                    id_to_dihedral_indices[torsion_id],
                    grid_id,
                    coordinates,
                    energy,
                )
                for (torsion_id, grid_id, coordinates, energy) in db.db.query(
                    DBQMTorsionPointRecord.parent_id,
                    DBQMTorsionPointRecord.grid_id,
                    DBQMTorsionPointRecord.coordinates,
                    DBQMTorsionPointRecord.energy,
                )
                .filter(DBQMTorsionPointRecord.parent_id.in_(ids_to_minimize))
                .all()
            )

        LOGGER.info(f"Passing generator of data to minimization with {force_field=}")

        minimization_results = _minimize_torsions(
            data=data,
            force_field=force_field,
            method=method,
            n_processes=n_processes,
            chunksize=chunksize,
            restraint_k=restraint_k,
        )

        LOGGER.info(f"Storing minimization results in database with {force_field=}")

        with self._get_session() as db:
            for result in minimization_results:
                db.store_mm_torsion_point(
                    MMTorsionPointRecord(
                        torsion_id=result.torsion_id,
                        grid_id=result.grid_id,
                        coordinates=result.coordinates,
                        force_field=result.force_field,
                        energy=result.energy,
                    ),
                )

    def get_rmsd(
        self,
        force_field: str,
        torsion_ids: list[int] | None = None,
        skip_check: bool = False,
        restraint_k: float = 0.0,
    ) -> RMSDCollection:
        """Get the RMSD summed over the torsion profile."""
        if not torsion_ids:
            torsion_ids = self.get_torsion_ids()

        if not skip_check:
            # TODO: Copy this into each get_* method?
            LOGGER.info("Calling optimize_mm from inside of get_log_sse.")
            self.optimize_mm(force_field=force_field, restraint_k=restraint_k)

        rmsds = RMSDCollection()

        for torsion_id in torsion_ids:
            qm_points = self.get_qm_points_by_torsion_id(torsion_id=torsion_id)
            mm_points = self.get_mm_points_by_torsion_id(torsion_id=torsion_id, force_field=force_field)

            molecule = Molecule.from_mapped_smiles(
                self.get_smiles_by_torsion_id(torsion_id),
                allow_undefined_stereo=True,
            )

            rmsds.append(RMSD.from_data(torsion_id, molecule, qm_points, mm_points))

        return rmsds

    def _get_energy_based_metric(
        self,
        force_field: str,
        analysis_metric_collection: type[AnalysisMetricCollectionTypeVar],
        torsion_ids: list[int] | None = None,
        skip_check: bool = False,
        restraint_k: float = 0.0,
        kwargs: dict | None = None,
    ) -> AnalysisMetricCollectionTypeVar:
        """Calculate energy-based metrics for the supplied analysis metric collection."""
        kwargs = kwargs if kwargs else dict()

        if not torsion_ids:
            torsion_ids = self.get_torsion_ids()

        if not skip_check:
            self.optimize_mm(force_field=force_field, restraint_k=restraint_k)

        collection = analysis_metric_collection()

        for torsion_id in torsion_ids:
            qm, mm = (
                numpy.fromiter(dct.values(), dtype=float)
                for dct in _normalize(
                    self.get_qm_energies_by_torsion_id(torsion_id=torsion_id),
                    self.get_mm_energies_by_torsion_id(torsion_id=torsion_id, force_field=force_field),
                )
            )

            if len(mm) * len(qm) == 0:
                LOGGER.warning(
                    "Missing QM OR MM data for this no mm data, returning empty dicts; \n\t"
                    f"{torsion_id=}, {force_field=}, {len(qm)=}, {len(mm)=}",
                )

            collection.append(
                collection.get_item_type().from_data(
                    torsion_id=torsion_id,
                    qm_energies=qm,
                    mm_energies=mm,
                    **kwargs,
                ),
            )

        return collection

    def get_rmse(
        self,
        force_field: str,
        torsion_ids: list[int] | None = None,
        skip_check: bool = False,
    ) -> RMSECollection:
        """Get the RMS RMSD over the torsion profile."""
        return self._get_energy_based_metric(
            force_field=force_field,
            analysis_metric_collection=RMSECollection,
            torsion_ids=torsion_ids,
            skip_check=skip_check,
        )

    def get_mean_error(
        self,
        force_field: str,
        torsion_ids: list[int] | None = None,
        skip_check: bool = False,
    ) -> MeanErrorCollection:
        return self._get_energy_based_metric(
            force_field=force_field,
            analysis_metric_collection=MeanErrorCollection,
            torsion_ids=torsion_ids,
            skip_check=skip_check,
        )

    def get_js_distance(
        self,
        force_field: str,
        torsion_ids: list[int] | None = None,
        skip_check: bool = False,
        temperature: float = 500.0,
    ) -> JSDistanceCollection:
        """Get the RMS RMSD over the torsion profile."""
        return self._get_energy_based_metric(
            force_field=force_field,
            analysis_metric_collection=JSDistanceCollection,
            torsion_ids=torsion_ids,
            skip_check=skip_check,
            kwargs={"temperature": temperature},
        )

    def get_outputs(self) -> MinimizedTorsionDataset:
        from yammbs.torsion.outputs import MinimizedTorsionProfile

        output_dataset = MinimizedTorsionDataset()

        LOGGER.info("Getting outputs for all force fields.")

        with self._get_session() as db:
            for force_field in self.get_force_fields():
                output_dataset.mm_torsions[force_field] = list()

                for torsion_id in self.get_torsion_ids():
                    mm_data = tuple(
                        (grid_id, coordinates, energy)
                        for (grid_id, coordinates, energy) in db.db.query(
                            DBMMTorsionPointRecord.grid_id,
                            DBMMTorsionPointRecord.coordinates,
                            DBMMTorsionPointRecord.energy,
                        )
                        .filter_by(parent_id=torsion_id)
                        .filter_by(force_field=force_field)
                        .all()
                    )

                    if len(mm_data) == 0:
                        continue

                    # TODO: Call _normalize here?
                    output_dataset.mm_torsions[force_field].append(
                        MinimizedTorsionProfile(
                            mapped_smiles=self.get_smiles_by_torsion_id(torsion_id),
                            dihedral_indices=self.get_dihedral_indices_by_torsion_id(torsion_id),
                            coordinates={grid_id: coordinates for grid_id, coordinates, _ in mm_data},
                            energies={grid_id: energy for grid_id, _, energy in mm_data},
                        ),
                    )

        return output_dataset

    def get_metrics(
        self,
        force_fields: Iterable[str] | None = None,
        js_temperature: float = 500.0,
        restraint_k: float = 0.0,
        skip_check: bool = True,
    ) -> MetricCollection:
        """Automatically compute all registered metrics for all force fields.

        Parameters
        ----------
        force_fields : Iterable[str] | None
            Iterable of force fields to compute metrics for. If None, compute for all available.
        js_temperature : float
            Temperature for JS distance calculation (default: 500.0 K).
        restraint_k : float
            Restraint force constant in kcal/(mol*Angstrom^2) for atoms not in dihedral.
            This is ignored if skip_check is True.
        skip_check : bool
            If True, skip the internal call to optimize_mm (assumes that the optimization has
            already been performed and ignores restraint_k).

        Returns
        -------
        MetricCollection
            A MetricCollection containing all computed metrics.

        """
        import pandas

        LOGGER.info("Getting metrics for all force fields.")

        metrics = MetricCollection()

        force_fields = force_fields if force_fields else self.get_force_fields()

        if not skip_check:
            LOGGER.info("Calling optimize_mm from inside of get_metrics.")
            for force_field in force_fields:
                self.optimize_mm(force_field=force_field, restraint_k=restraint_k)

        # TODO: Optimize this for speed
        for force_field in force_fields:
            rmses = self.get_rmse(force_field=force_field, skip_check=True).to_dataframe()
            rmsds = self.get_rmsd(force_field=force_field, skip_check=True).to_dataframe()
            mean_errors = self.get_mean_error(force_field=force_field, skip_check=True).to_dataframe()
            js_distances = self.get_js_distance(
                force_field=force_field,
                skip_check=True,
                temperature=js_temperature,
            ).to_dataframe()

            dataframe = rmses.join(rmsds).join(mean_errors).join(js_distances)

            dataframe = dataframe.replace({pandas.NA: numpy.nan})

            metrics.metrics[force_field] = {
                id: Metric(  # type: ignore[misc]
                    rmsd=row["rmsd"],
                    rmse=row["rmse"],
                    mean_error=row["mean_error"],
                    js_distance=(row["js_distance"], row["js_temperature"]),
                )
                for id, row in dataframe.iterrows()
            }

        return metrics

    def get_proper_torsion_parameters_by_torsion_id(
        self,
        torsion_id: int,
        force_field_name: str,
    ) -> list[ProperTorsionHandler.ProperTorsionType]:
        """Get the proper torsion parameters which match the dihedral being scanned."""
        # Get the central two atoms for the dihedral being scanned
        central_dihedral_indices = set(self.get_dihedral_indices_by_torsion_id(torsion_id)[1:3])
        ff = _lazy_load_force_field(force_field_name)
        mol = Molecule.from_mapped_smiles(
            self.get_smiles_by_torsion_id(torsion_id),
            allow_undefined_stereo=True,
        )
        proper_torsions = ff.label_molecules(mol.to_topology())[0]["ProperTorsions"]

        # Get all dihedrals which match the central two atoms
        matched_dihedrals = []
        for indices, dihedral in proper_torsions.items():
            # Make sure we're independent of atom ordering
            if set(indices[1:3]) == set(central_dihedral_indices):
                matched_dihedrals.append(dihedral)

        return matched_dihedrals

    def get_torsion_image(self, torsion_id: int) -> str:
        """Get an image of the molecule with the dihedral highlighted."""
        import base64

        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D

        smiles = self.get_smiles_by_torsion_id(torsion_id)
        dihedral_indices = self.get_dihedral_indices_by_torsion_id(torsion_id)

        # Use the mapped SMILES to get the molecule
        mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
        if mol is None:
            raise ValueError(f"Could not convert SMILES to molecule: {smiles}")

        rdmol = mol.to_rdkit()

        # Draw in 2D - compute 2D coordinates
        AllChem.Compute2DCoords(rdmol)  # type: ignore[attr-defined, unused-ignore]
        # Highlight the dihedral
        atom_indices = [
            dihedral_indices[0],
            dihedral_indices[1],
            dihedral_indices[2],
            dihedral_indices[3],
        ]
        bond_indices = [
            rdmol.GetBondBetweenAtoms(atom_indices[0], atom_indices[1]).GetIdx(),
            rdmol.GetBondBetweenAtoms(atom_indices[1], atom_indices[2]).GetIdx(),
            rdmol.GetBondBetweenAtoms(atom_indices[2], atom_indices[3]).GetIdx(),
        ]

        # Create an SVG drawer
        drawer = rdMolDraw2D.MolDraw2DSVG(200, 200)  # Set the size of the image (width x height)
        drawer.SetFontSize(0.8)  # Optional: Adjust font size for better readability

        # Prepare the molecule for drawing
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            rdmol,
            highlightAtoms=atom_indices,
            highlightBonds=bond_indices,
        )

        # Finish the drawing and get the SVG text
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        svg_base64 = base64.b64encode(svg.encode()).decode()
        img_tag = f'<img src="data:image/svg+xml;base64,{svg_base64}" alt="Molecule Image" />'
        return img_tag

    def get_scan_image(
        self,
        torsion_id: int,
        force_fields: list[str],
    ) -> str:
        from matplotlib import pyplot as plt

        # Create plot with two subplots
        fig, (energy_axis, geometry_axis) = plt.subplots(
            2,
            1,
            figsize=(3.6, 3.2),
            dpi=300,
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1]},
        )

        # Get the energies
        _qm = self.get_qm_energies_by_torsion_id(torsion_id)
        _qm = dict(sorted(_qm.items()))
        qm_minimum_index = min(_qm, key=_qm.get)  # type: ignore[arg-type]

        # Make a new dict to avoid in-place modification while iterating
        qm = {key: _qm[key] - _qm[qm_minimum_index] for key in _qm}

        # Get QM and MM points for RMSD calculation
        qm_points = self.get_qm_points_by_torsion_id(torsion_id=torsion_id)
        molecule = Molecule.from_mapped_smiles(
            self.get_smiles_by_torsion_id(torsion_id),
            allow_undefined_stereo=True,
        )

        # Ensure colors are not reused across force fields - use a colour map
        # with 10 colours and change the symbol for each if more than 10 force fields
        cmap = plt.get_cmap("tab10")
        symbols = ["o", "^", "s"]  # Allow up to 30 force fields

        for i, force_field in enumerate(force_fields):
            mm = dict(sorted(self.get_mm_energies_by_torsion_id(torsion_id, force_field=force_field).items()))
            assert mm.keys() == qm.keys(), "MM data and QM data should have the same keys"
            if len(mm) == 0:
                continue

            color = cmap(i % 10)
            marker = symbols[i // 10]

            # Plot energies
            energy_axis.plot(
                list(mm.keys()),
                [val - mm[qm_minimum_index] for val in mm.values()],
                label=force_field,
                color=color,
                marker=marker,
            )

            # Calculate and plot RMSD at each point
            mm_points = self.get_mm_points_by_torsion_id(
                torsion_id=torsion_id,
                force_field=force_field,
            )

            angles = []
            rmsds = []
            for angle in sorted(mm_points.keys()):
                if angle in qm_points:
                    qm_coords = qm_points[angle]
                    mm_coords = mm_points[angle]
                    # Calculate RMSD for this point
                    rmsd_value = RMSD.from_data(
                        torsion_id=torsion_id,
                        molecule=molecule,
                        qm_points={angle: qm_coords},
                        mm_points={angle: mm_coords},
                    ).rmsd
                    angles.append(angle)
                    rmsds.append(rmsd_value)

            geometry_axis.plot(
                angles,
                rmsds,
                label=force_field,
                color=color,
                marker=marker,
            )

        energy_axis.plot(
            list(qm.keys()),
            list(qm.values()),
            "k.-",
            label="QM",
        )

        energy_axis.legend(loc=0, bbox_to_anchor=(1.05, 1))

        # Label the axes
        energy_axis.set_ylabel(r"Energy / kcal mol$^{-1}$")
        geometry_axis.set_ylabel(r"RMSD / $\mathrm{\AA}$")
        geometry_axis.set_xlabel("Torsion angle / degrees")

        # Convert the plot to SVG
        import base64
        from io import BytesIO

        buf = BytesIO()
        plt.savefig(buf, format="svg", bbox_inches="tight")
        buf.seek(0)
        svg_data = buf.getvalue()
        buf.close()

        # Close the figure
        plt.close(fig)

        # Encode the SVG data to base64
        svg_base64 = base64.b64encode(svg_data).decode()
        img_tag = f'<img src="data:image/svg+xml;base64,{svg_base64}" alt="Scan Image" />'
        return img_tag

    def get_summary_df(
        self,
        force_fields: list[str],
        show_parameters: bool = False,
    ) -> pd.DataFrame:
        """Get a summary dataframe of the metrics for a given force field.

        This is intended to be used for generating HTML reports.

        Parameters
        ----------
        force_fields : list[str]
            The force fields to include in the summary dataframe.

        show_parameters : bool
            Whether to include the dihedral parameters in the summary dataframe.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the summary of the metrics for the given force fields.

        """
        import pandas as pd
        from tqdm import tqdm

        rows = []
        metrics = self.get_metrics().metrics
        metrics_to_plot = {
            "RMSD / A": lambda x: x.rmsd,
            "RMSE / kcal mol-1": lambda x: x.rmse,
            "Mean Error / kcal mol-1": lambda x: x.mean_error,
            "JS Distance": lambda x: x.js_distance[0],
        }

        for mol_id in tqdm(self.get_torsion_ids()):
            row: dict[str, str | float] = {}
            row["ID"] = mol_id
            row["Torsion Image"] = self.get_torsion_image(mol_id)
            row["Scan Image"] = self.get_scan_image(mol_id, force_fields=force_fields)

            for metric, metric_func in metrics_to_plot.items():
                for force_field in force_fields:
                    row[f"{metric}\n{force_field}"] = metric_func(metrics[force_field][mol_id])

            # Also add the dihedral type for each force field, if requested
            if show_parameters:
                for force_field in force_fields:
                    proper_torsions = self.get_proper_torsion_parameters_by_torsion_id(
                        torsion_id=mol_id,
                        force_field_name=force_field,
                    )
                    row[f"Proper Torsion SMIRKS\n{force_field}"] = "\n".join(t.smirks for t in proper_torsions)
                    row[f"Proper Torsion\n{force_field}"] = "\n".join(str(t) for t in proper_torsions)

            rows.append(row)

        return pd.DataFrame(rows)

    def get_summary(
        self,
        file_name: str,
        force_fields: list[str] | None = None,
        show_parameters: bool = False,
    ) -> None:
        """Create a html summary table of the metrics for a given force field.

        Parameters
        ----------
        file_name : str
            The name of the file to save the summary to.
        force_fields : list[str] | None, optional
            The force fields to include in the summary. If None, include all force fields.
        show_parameters : bool, optional
            Whether to include the dihedral parameters in the summary. This is False by default
            as it substantially slows down the generation of the summary.

        Returns
        -------
        None

        """
        import bokeh
        import panel

        force_fields = force_fields if force_fields else self.get_force_fields()

        df = self.get_summary_df(force_fields, show_parameters=show_parameters)

        number_format = bokeh.models.widgets.tables.NumberFormatter(format="0.0000")
        string_format = {"type": "textarea", "whiteSpace": "pre-wrap"}

        formatters: dict[str, bokeh.models.widgets.tables.NumberFormatter | dict[str, str] | str] = {}
        for col in df.columns:
            if "Image" in col:
                formatters[col] = "html"
            elif "Proper Torsion" in col:
                formatters[col] = string_format
            else:
                formatters[col] = number_format

        frozen_colums = ["ID", "Torsion Image", "Scan Image"]

        # Scale up the row height depending on the number of force fields shown
        row_height = max(300, 25 * len(force_fields))
        n_rows = 800 // row_height

        tabulator = panel.widgets.Tabulator(
            df,
            show_index=False,
            selectable=False,
            disabled=True,
            formatters=formatters,
            configuration={"rowHeight": row_height},
            sizing_mode="stretch_width",
            frozen_columns=frozen_colums,
            page_size=n_rows,
            pagination="local",
        )

        # TODO: Colour scale the metrics

        layout = panel.Column(
            None,
            tabulator,
        )

        layout.save(file_name, title="MetricsSummary", embed=True)
