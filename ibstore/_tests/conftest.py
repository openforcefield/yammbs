import pytest
from openff.qcsubmit.results import OptimizationResultCollection
from openff.utilities.utilities import get_data_file_path


@pytest.fixture()
def small_collection() -> OptimizationResultCollection:
    return OptimizationResultCollection.parse_file(
        get_data_file_path("_tests/data/01-processed-qm-ch.json", "ibstore")
    )
