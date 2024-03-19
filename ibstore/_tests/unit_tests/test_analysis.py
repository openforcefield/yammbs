import pandas
import pytest
from openff.toolkit import Molecule

from ibstore.analysis import get_internal_coordinate_rmsds, get_rmsd


class TestAnalysis:
    def test_rmsd(self, allicin):
        allicin.generate_conformers(n_conformers=10)

        # Passing the same conformers should return 0.0
        last_last = get_rmsd(
            molecule=allicin,
            reference=allicin.conformers[-1],
            target=allicin.conformers[-1],
        )

        assert last_last == 0.0

        first_last = get_rmsd(
            molecule=allicin,
            reference=allicin.conformers[0],
            target=allicin.conformers[-1],
        )

        assert isinstance(first_last, float)

        first_second = get_rmsd(
            molecule=allicin,
            reference=allicin.conformers[0],
            target=allicin.conformers[1],
        )

        assert first_second != first_last

        last_first = get_rmsd(
            molecule=allicin,
            reference=allicin.conformers[-1],
            target=allicin.conformers[0],
        )
        assert last_first == first_last


class TestInternalCoordinateRMSD:
    def test_rmsds_between_conformers(self, ligand):
        assert ligand.n_conformers

        rmsds = get_internal_coordinate_rmsds(
            molecule=ligand,
            reference=ligand.conformers[0],
            target=ligand.conformers[-1],
        )

        assert all(
            [rmsds[key] > 0.0 for key in ["Bond", "Angle", "Dihedral", "Improper"]],
        )

    def test_matching_conformers_zero_rmsd(self, ligand):
        rmsds = get_internal_coordinate_rmsds(
            molecule=ligand,
            reference=ligand.conformers[0],
            target=ligand.conformers[0],
        )

        assert all(
            [rmsds[key] == 0.0 for key in ["Bond", "Angle", "Dihedral", "Improper"]],
        )

    def test_no_torsions(self, water):
        rmsds = get_internal_coordinate_rmsds(
            molecule=water,
            reference=water.conformers[0],
            target=water.conformers[0],
        )

        assert rmsds["Bond"] == 0.0
        assert rmsds["Angle"] == 0.0

        assert "Dihedral" not in rmsds
        assert "Improper" not in rmsds

    def test_no_impropers(self, hydrogen_peroxide):
        rmsds = get_internal_coordinate_rmsds(
            molecule=hydrogen_peroxide,
            reference=hydrogen_peroxide.conformers[0],
            target=hydrogen_peroxide.conformers[0],
        )

        assert "Dihedral" in rmsds
        assert "Improper" not in rmsds

    def test_icrmsd_dataframe(self, small_store):
        dataframe = small_store.get_internal_coordinate_rmsd(
            "openff-2.0.0",
        ).to_dataframe()

        cumene_id = small_store.get_molecule_id_by_qcarchive_id(37017037)

        # This molecule should be cumene
        assert Molecule.from_mapped_smiles(
            small_store.get_smiles_by_molecule_id(cumene_id),
        ).is_isomorphic_with(Molecule.from_smiles("CC(C)c1ccccc1"))

        for qcarchive_id in small_store.get_qcarchive_ids_by_molecule_id(cumene_id):
            # This should probably be an int, but it's str
            data = dataframe.loc[int(qcarchive_id)]

            for key in ("Bond", "Angle", "Dihedral", "Improper"):
                assert isinstance(data[key], float)
                assert data[key] != pytest.approx(0.0)

    def test_torsions_not_in_methane_icrmsd(self, small_store):
        dataframe = small_store.get_internal_coordinate_rmsd(
            "openff-2.0.0",
        ).to_dataframe()

        methane_id = small_store.get_molecule_id_by_qcarchive_id(37017014)

        # This molecule should be methane
        assert Molecule.from_mapped_smiles(
            small_store.get_smiles_by_molecule_id(methane_id),
        ).is_isomorphic_with(Molecule.from_smiles("C"))

        for qcarchive_id in small_store.get_qcarchive_ids_by_molecule_id(methane_id):
            # This should probably be an int, but it's str
            data = dataframe.loc[int(qcarchive_id)]

            assert isinstance(data["Bond"], float)
            assert isinstance(data["Angle"], float)
            assert data["Dihedral"] is pandas.NA
            assert data["Improper"] is pandas.NA
