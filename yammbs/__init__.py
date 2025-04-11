"""YAMMBS: Yet Another Molecular Mechanics Benchmarking Suite."""

from importlib.metadata import version

from yammbs._store import MoleculeStore

__all__ = ("MoleculeStore",)

__version__ = version("yammbs")
