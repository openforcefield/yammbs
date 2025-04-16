"""Analysis methods for torsion profiles."""

import logging
from abc import ABC
from typing import TYPE_CHECKING, Generic, Self, TypeVar

import numpy

from yammbs._base.array import Array
from yammbs._base.base import ImmutableModel
from yammbs.analysis import get_rmsd

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas
    from openff.toolkit import Molecule


def _normalize(qm: dict[float, float], mm: dict[float, float]) -> tuple[dict[float, float], dict[float, float]]:
    """Normalize, after sorting, a pair of QM and MM profiles to the values at the QM minimum."""
    if len(mm) == 0:
        LOGGER.warning(
            f"no mm data, returning empty dicts; length of qm dict is {len(qm)=}",
        )
        return dict(), dict()

    _qm = dict(sorted(qm.items()))
    qm_minimum_index = min(_qm, key=_qm.get)  # type: ignore[arg-type]

    return {key: _qm[key] - _qm[qm_minimum_index] for key in _qm}, {
        key: mm[key] - mm[qm_minimum_index] for key in sorted(mm)
    }


_ImmutableModelTypeVar = TypeVar("_ImmutableModelTypeVar", bound=ImmutableModel)


class AnalysisMetricCollection(ABC, Generic[_ImmutableModelTypeVar], list[_ImmutableModelTypeVar]):
    """A generic collection class for typed lists of analysis metrics."""

    item_type: type[_ImmutableModelTypeVar]  # This must be set in subclasses

    @classmethod
    def get_item_type(cls) -> type[_ImmutableModelTypeVar]:
        """Retrieve the type of items in the collection (_ImmutableModelTypeVar)."""
        if not hasattr(cls, "item_type") or cls.item_type is None:
            raise NotImplementedError(f"{cls.__name__} must define the 'item_type' class variable.")
        return cls.item_type

    def to_dataframe(self) -> "pandas.DataFrame":
        """Convert the collection to a pandas DataFrame."""
        import pandas

        item_type = self.get_item_type()
        return pandas.DataFrame(
            [element.__dict__ for element in self],
            index=pandas.Index([element.id for element in self]),
            columns=[field for field in item_type.model_fields.keys() if field != "id"],
        )

    def to_csv(self, path: str):
        """Save the collection to a CSV file."""
        self.to_dataframe().to_csv(path)


class Minima(ImmutableModel):
    """A model storing the minima of a torsion profile."""

    id: int
    minima: list[float]


class RMSD(ImmutableModel):
    """A model storing the RMS RMSD over a torsion profile."""

    id: int
    rmsd: float

    @classmethod
    def from_data(cls, molecule_id: int, molecule: "Molecule", qm_points: Array, mm_points: Array) -> Self:
        """Create an RMSD object by calculating the RMSD between QM and MM points."""
        rmsd_vals = numpy.array([get_rmsd(molecule, qm_points[key], mm_points[key]) for key in qm_points])
        return cls(
            id=molecule_id,
            rmsd=numpy.sqrt((rmsd_vals**2).mean()),
        )


class RMSDCollection(AnalysisMetricCollection[RMSD]):
    """A collection of RMSD models."""

    item_type = RMSD


class RMSE(ImmutableModel):
    """A model storing the RMSE error over a torsion profile."""

    id: int
    rmse: float

    @classmethod
    def from_data(cls, molecule_id: int, qm_energies: Array, mm_energies: Array) -> Self:
        """Create an RMSE object by calculating the RMSE between QM and MM energies."""
        return cls(
            id=molecule_id,
            rmse=numpy.sqrt(((qm_energies - mm_energies) ** 2).mean()),
        )


class RMSECollection(AnalysisMetricCollection[RMSE]):
    """A collection of RMSE models."""

    item_type = RMSE


class MeanError(ImmutableModel):
    """A model storing the mean error over a torsion profile."""

    id: int
    mean_error: float

    @classmethod
    def from_data(cls, molecule_id: int, qm_energies: Array, mm_energies: Array) -> Self:
        """Create a MeanError object by calculating the mean MM - QM energy."""
        return cls(
            id=molecule_id,
            mean_error=numpy.mean(mm_energies - qm_energies),
        )


class MeanErrorCollection(AnalysisMetricCollection[MeanError]):
    """A collection of MeanError models."""

    item_type = MeanError


class JSDistance(ImmutableModel):
    """A model storing the Jensen-Shannon distances."""

    id: int
    js_distance: float
    js_temperature: float

    @classmethod
    def from_data(
        cls,
        molecule_id: int,
        qm_energies: Array,
        mm_energies: Array,
        temperature: float,
    ) -> Self:
        """Create a JSDistance object from supplied data.

        Do this by calculating the Jensen-Shannon distance
        between the two distributions generated with Boltzmann inversion at the
        specified temperature, using base 2 logs. A distance of 0 indicates
        identical distributions, while a distance of 1 indicates completely
        non-overlapping distributions.
        """
        from scipy.spatial.distance import jensenshannon

        beta = 1.0 / (temperature * 0.0019872043)  # kcal/mol to K

        # Get normalised probabilities by Boltzmann inversion
        p_qm, p_mm = numpy.exp(-beta * qm_energies), numpy.exp(-beta * mm_energies)
        p_qm /= p_qm.sum()
        p_mm /= p_mm.sum()

        return cls(
            id=molecule_id,
            # Use base 2 so that the upper limit is 1
            js_distance=jensenshannon(p_qm, p_mm, base=2),
            js_temperature=temperature,
        )


class JSDistanceCollection(AnalysisMetricCollection[JSDistance]):
    """A collection of JSDistance models."""

    item_type = JSDistance
