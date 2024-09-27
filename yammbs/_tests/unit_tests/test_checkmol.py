import pytest
from openff.toolkit import Molecule
from openff.utilities import has_executable

from yammbs.checkmol import ChemicalEnvironment, analyze_functional_groups


@pytest.mark.skipif(not has_executable("checkmol"), reason="checkmol not installed")
@pytest.mark.parametrize(
    "mapped",
    [True, False],
)
def test_analyze_ethanol(mapped):
    smiles = "CCO"

    if mapped:
        smiles = Molecule.from_smiles("CCO").to_smiles(mapped=True)

    groups = analyze_functional_groups(smiles)

    assert len(groups) == 3

    for group in (
        ChemicalEnvironment.Hydroxy,
        ChemicalEnvironment.Alcohol,
        ChemicalEnvironment.PrimaryAlcohol,
    ):
        assert group in groups
