"""YAMMBS: Yet Another Molecular Mechanics Benchmarking Suite."""

import multiprocessing
from importlib.metadata import version

from yammbs._store import MoleculeStore

__all__ = ("MoleculeStore",)

__version__ = version("yammbs")

multiprocessing.get_start_method("spawn")
