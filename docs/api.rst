API Reference
=============

Below is the API reference for ``yammbs``. See the `examples <examples.html>`_ for details on how to use these objects.

Analysis
--------

Tools for analyzing molecular mechanics results and comparing with QM data.

.. currentmodule:: yammbs.analysis
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    DDE
    DDECollection
    ICRMSD
    ICRMSDCollection
    RMSD
    RMSDCollection
    TFD
    TFDCollection
    get_internal_coordinate_differences
    get_internal_coordinate_rmsds
    get_internal_coordinates
    get_rmsd
    get_tfd

Checkmol
--------

Molecular structure analysis using the checkmol utility.

.. currentmodule:: yammbs.checkmol
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    analyze_functional_groups
    ChemicalEnvironment

Exceptions
----------

Custom exceptions for YAMMBS.

.. currentmodule:: yammbs.exceptions
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    DatabaseExistsError

Inputs
------

Input data models for reading QCArchive and QM datasets.

.. currentmodule:: yammbs.inputs
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    QCArchiveDataset
    QCArchiveMolecule
    QMDataset
    QMMolecule

Models
------

Core data models for molecular mechanics calculations.

.. currentmodule:: yammbs.models
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    MoleculeRecord
    QMConformerRecord
    MMConformerRecord
    Record

Outputs
-------

Output data models for storing and analyzing results.

.. currentmodule:: yammbs.outputs
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    Metric
    MetricCollection
    MinimizedMolecule
    MinimizedQCArchiveDataset
    MinimizedQCArchiveMolecule
    MinimizedQMDataset
