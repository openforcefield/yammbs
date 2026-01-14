"""Integration test for run_torsion_comparisons.py script."""

import os
import pathlib

import pytest
from click.testing import CliRunner
from openff.utilities import get_data_file_path

from yammbs.scripts.run_torsion_comparisons import main

# Set to True to keep generated figures for manual inspection
_KEEP_FIGURES = os.environ.get("YAMMBS_KEEP_TEST_FIGURES", "").lower() in (
    "1",
    "true",
    "yes",
)


@pytest.fixture
def torsion_input_file():
    """Return path to torsiondrive test data."""
    return get_data_file_path(
        "_tests/data/yammbs/rowley-biaryl-torsiondrive-data.json",
        "yammbs",
    )


@pytest.fixture
def test_database(tmp_path):
    """Copy the pre-populated test database to a temporary location."""
    import shutil

    source_db = get_data_file_path(
        "_tests/data/yammbs/rowley-biaryl-torsiondrive-data.sqlite",
        "yammbs",
    )

    # Copy to temp directory so tests don't modify the original
    dest_db = tmp_path / "test-torsion.sqlite"
    shutil.copy(source_db, dest_db)

    return dest_db


def test_run_torsion_comparisons_integration(test_database, torsion_input_file, tmp_path):
    """Test that run_torsion_comparisons.py runs end-to-end with pre-minimized inputs.

    This is a fast integration test that:
    - Uses existing test data (20 torsion profiles)
    - Uses a pre-populated database with openff-2.2.1 minimized structures
    - Verifies that all expected output files are created
    - Does not validate the correctness of the outputs (that's for unit tests)

    To keep generated figures for inspection, set the environment variable:
        export YAMMBS_KEEP_TEST_FIGURES=1
    """
    db_file = test_database

    # Setup paths in temporary directory
    metrics_file = tmp_path / "metrics.json"
    minimized_file = tmp_path / "minimized.json"

    if _KEEP_FIGURES:
        # Use a persistent directory for figures
        plot_dir = pathlib.Path("test_torsion_plots_output")
        plot_dir.mkdir(exist_ok=True)
        print(f"\nKeeping test figures in: {plot_dir.absolute()}")
    else:
        plot_dir = tmp_path / "plots"
        plot_dir.mkdir()

    # Run the CLI command - since we have pre-minimized data, this should be fast
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--qcarchive-torsion-data",
            torsion_input_file,
            "--base-force-fields",
            "openff-1.0.0",  # Use two pre-minimized force fields
            "--base-force-fields",
            "openff-2.2.1",
            "--database-file",
            str(db_file),
            "--output-metrics",
            str(metrics_file),
            "--output-minimized",
            str(minimized_file),
            "--plot-dir",
            str(plot_dir),
        ],
    )

    # Check that the command completed successfully
    assert result.exit_code == 0, f"Command failed with output: {result.output}"

    # Verify that expected output files were created
    assert db_file.exists(), "Database file was not created"
    assert metrics_file.exists(), "Metrics file was not created"
    assert minimized_file.exists(), "Minimized file was not created"

    # Verify that plot files were created
    expected_plots = [
        "torsions.png",
        "rms_rmsd.png",
        "rmse.png",
        "js_distance.png",
        "rms_rmsd_rms.png",
        "rmse_rms.png",
        "mean_js_distance.png",
        "mean_error_distribution.png",
    ]
    for plot_name in expected_plots:
        plot_path = plot_dir / plot_name
        assert plot_path.exists(), f"Expected plot file {plot_name} was not created"

    # Basic sanity checks on file sizes (should not be empty)
    assert db_file.stat().st_size > 0, "Database file is empty"
    assert metrics_file.stat().st_size > 100, "Metrics file is too small or empty"
    assert minimized_file.stat().st_size > 100, "Minimized file is too small or empty"
