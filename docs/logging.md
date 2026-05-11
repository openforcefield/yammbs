# Logging in YAMMBS

Developer summary of what is logged.

## Logger behavior

- Modules use `logging.getLogger(__name__)`.
- YAMMBS does not configure root logging.
- Runtime package code currently does not emit `ERROR` / `EXCEPTION` / `CRITICAL`.

## Core minimization modules

### `yammbs/_minimize.py`

- `DEBUG`
  - Initial energy.
  - Initial positions.
  - Final energy.
  - Per-record completion (with `qcarchive_id` and final energy).
- `WARNING`
  - Record skipped due to unassigned valence.
  - Record skipped due to `ValueError` during setup.

### `yammbs/torsion/_minimize.py`

- `DEBUG`
  - Input mapping/pool setup for torsion minimization.
  - Setup details (method, input payload, molecule/system creation).
  - Restraint construction details.
  - Initial/final dihedral diagnostics.
  - Initial/final energies (excluding restraint).
  - Per-record completion (with `torsion_id` and final energy).
- `WARNING`
  - Record skipped due to unassigned valence.
  - Record skipped due to setup `RuntimeError` / `ValueError`.

## Other modules

- `yammbs/_forcefields.py`: `INFO` for SMIRNOFF constraint deregistration.
- `yammbs/_store.py`: `WARNING` when TFD calculation fails.
- `yammbs/torsion/inputs.py`: `INFO` on QCSubmit-to-YAMMBS conversion.
- `yammbs/torsion/analysis.py`: `WARNING` for missing MM data.
- `yammbs/torsion/_store.py`: `INFO` for torsion workflow stages; `WARNING` for missing QM/MM data.
