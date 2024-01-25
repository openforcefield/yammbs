"""
def get_internal_coordinate_rmsds(
    molecule: Molecule,
    reference: Array,
    target: Array,
    _types: tuple[str] = ("Bond", "Angle", "Dihedral", "Improper"),
) -> dict[str, float]:
"""
from ibstore.analysis import get_internal_coordinate_rmsds


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
