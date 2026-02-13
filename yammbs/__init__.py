"""YAMMBS: Yet Another Molecular Mechanics Benchmarking Suite."""

import os

# Force single-threaded execution for numpy
# to ensure good performance for geomeTRIC
# Must be set before importing numpy!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from importlib.metadata import version

from yammbs._store import MoleculeStore

__all__ = ("MoleculeStore",)

__version__ = version("yammbs")
