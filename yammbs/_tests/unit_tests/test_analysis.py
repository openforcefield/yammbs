import numpy
import pandas
from openff.utilities import get_data_file_path
import pytest
from openff.toolkit import ForceField, Molecule

from yammbs.analysis import get_internal_coordinate_differences, get_internal_coordinate_rmsds, get_rmsd, get_tfd, get_internal_coordinates


class TestAnalysis:
    def test_rmsd(self, allicin, conformers):
        # Passing the same conformers should return 0.0
        last_last = get_rmsd(
            molecule=allicin,
            reference=conformers[-1],
            target=conformers[-1],
        )

        assert last_last == 0.0

        first_last = get_rmsd(
            molecule=allicin,
            reference=conformers[0],
            target=conformers[-1],
        )

        assert isinstance(first_last, float)

        first_second = get_rmsd(
            molecule=allicin,
            reference=conformers[0],
            target=conformers[1],
        )

        assert first_second != first_last

        last_first = get_rmsd(
            molecule=allicin,
            reference=conformers[-1],
            target=conformers[0],
        )

        assert last_first == pytest.approx(first_last)

    def test_tfd(self, allicin, conformers):
        # Passing the same conformers should return 0.0
        last_last = get_tfd(
            molecule=allicin,
            reference=conformers[-1],
            target=conformers[-1],
        )

        assert last_last == 0.0

        first_last = get_tfd(
            molecule=allicin,
            reference=conformers[0],
            target=conformers[-1],
        )

        assert isinstance(first_last, float)

        first_second = get_tfd(
            molecule=allicin,
            reference=conformers[0],
            target=conformers[1],
        )

        assert first_second != first_last

        last_first = get_tfd(
            molecule=allicin,
            reference=conformers[-1],
            target=conformers[0],
        )

        assert last_first == first_last

class TestgeomeTRICEdgeCases:
    def test_linear_angles_not_missed(self):
        carbon_dioxide = Molecule.from_smiles("O=C=O")

        # this will probably only be one conformer, but that's fine,
        # we just want to make sure we get linear angles out at all
        carbon_dioxide.generate_conformers(n_conformers=1)
        internal_coordinates = get_internal_coordinates(
            molecule=carbon_dioxide,
            reference=carbon_dioxide.conformers[0],
            target=carbon_dioxide.conformers[0],
        )


        assert "Angle" in internal_coordinates
        assert len(internal_coordinates["Angle"]) == 1

        for val in internal_coordinates["Angle"].values():
            # if the two "conformers" are the same,
            # the measured angles should be the same
            assert (val[0] == val[1]) == (carbon_dioxide.n_conformers == 1)

class TestInternalCoordinateRMSD:
    def test_missing_type(self, ligand):
        with pytest.raises(ValueError, match="Urey"):
            get_internal_coordinate_rmsds(
                molecule=ligand,
                reference=ligand.conformers[0],
                target=ligand.conformers[-1],
                _types=["Bond", "Angle", "Dihedral", "Urey-Bradley"] ,
            )

    def test_rmsds_between_conformers(self, ligand):
        assert ligand.n_conformers > 1

        assert not numpy.allclose(
            ligand.conformers[0],
            ligand.conformers[-1],
        )

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

    def test_internal_coordinate_differences(self, small_store, guess_n_processes):
        small_store.optimize_mm(
            force_field="openff-2.0.0",
            n_processes=guess_n_processes,
        )

        molecule = Molecule.from_mapped_smiles(small_store.get_smiles_by_molecule_id(1))

        differences = get_internal_coordinate_differences(
            molecule=molecule,
            reference=small_store.get_qm_conformer_records_by_molecule_id(1)[-1].coordinates,
            target=small_store.get_mm_conformer_records_by_molecule_id(1, force_field="openff-2.0.0")[-1].coordinates,
        )

        assert len(differences) == 4

        assert len(differences["Bond"]) == molecule.n_bonds
        assert len(differences["Angle"]) <= molecule.n_angles  # can be different - not all angles are parameterized
        assert len(differences["Dihedral"]) == molecule.n_propers
        assert len(differences["Improper"]) <= molecule.n_impropers

        assert max(differences["Bond"].values()) < 0.1
        assert max(differences["Angle"].values()) < 1
        assert max(differences["Dihedral"].values()) < 10
        assert max(differences["Improper"].values()) < 1

    @pytest.mark.skip(reason="See https://github.com/openforcefield/yammbs/issues/199")
    def test_internal_coordinate_impropers(self):
        triazine = Molecule.from_mapped_smiles("[H:7][c:1]1[n:2][c:3]([n:4][c:5]([n:6]1)[H:9])[H:8]")

        triazine.generate_conformers(n_conformers=1)

        # these should have central atom (each carbon, atoms 0, 2, 4 in this mapped SMILES) listed SECOND
        sage_impropers: list[tuple[int, int, int, int]] = sorted(
            [*ForceField("openff-2.2.1.offxml").label_molecules(triazine.to_topology())][0]["ImproperTorsions"],
        )

        # these should also, by design, list the central atoms SECOND
        differences = get_internal_coordinate_differences(
            molecule=triazine,
            reference=triazine.conformers[0],
            target=triazine.conformers[0],
        )

        # should be [(1, 0, 5, 6), (1, 2, 3, 7), (3, 4, 5, 8)]
        assert sorted(differences["Improper"].keys()) == sage_impropers
