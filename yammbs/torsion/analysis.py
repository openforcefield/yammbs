"""Analysis methods for torsion profiles."""

import logging
from abc import ABC
from typing import TYPE_CHECKING, Generic, Self, TypeVar

import numpy

from yammbs._base.array import Array
from yammbs._base.base import ImmutableModel
from yammbs.analysis import get_rmsd, get_tfd

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


class AnalysisMetric(ABC, ImmutableModel):
    """A model storing a single analysis metric."""

    id: int


AnalysisMetricTypeVar = TypeVar("AnalysisMetricTypeVar", bound=AnalysisMetric)


class AnalysisMetricCollection(ABC, Generic[AnalysisMetricTypeVar], list[AnalysisMetricTypeVar]):
    """A generic collection class for typed lists of analysis metrics."""

    item_type: type[AnalysisMetricTypeVar]  # This must be set in subclasses

    @classmethod
    def get_item_type(cls) -> type[AnalysisMetricTypeVar]:
        """Retrieve the type of items in the collection (AnalysisMetricTypeVar)."""
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


AnalysisMetricCollectionTypeVar = TypeVar("AnalysisMetricCollectionTypeVar", bound=AnalysisMetricCollection)


class Minima(AnalysisMetric):
    """A model storing the minima of a torsion profile."""

    id: int
    minima: list[float]


class RMSRMSD(AnalysisMetric):
    """A model storing the RMS RMSD over a torsion profile."""

    id: int
    rms_rmsd: float

    @classmethod
    def from_data(
        cls,
        torsion_id: int,
        molecule: "Molecule",
        qm_points: dict[float, Array],
        mm_points: dict[float, Array],
    ) -> Self:
        """Create an RMSD object by calculating the RMSD between QM and MM points."""
        rmsd_vals = numpy.array([get_rmsd(molecule, qm_points[key], mm_points[key]) for key in qm_points])
        return cls(
            id=torsion_id,
            rms_rmsd=numpy.sqrt((rmsd_vals**2).mean()),
        )


class RMSRMSDCollection(AnalysisMetricCollection[RMSRMSD]):
    """A collection of RMSD models."""

    item_type = RMSRMSD


class MeanRMSD(AnalysisMetric):
    """A model storing the mean RMSD over a torsion profile."""

    id: int
    mean_rmsd: float

    @classmethod
    def from_data(
        cls,
        torsion_id: int,
        molecule: "Molecule",
        qm_points: dict[float, Array],
        mm_points: dict[float, Array],
    ) -> Self:
        """Create an RMSD object by calculating the RMSD between QM and MM points."""
        rmsd_vals = numpy.array([get_rmsd(molecule, qm_points[key], mm_points[key]) for key in qm_points])
        return cls(id=torsion_id, mean_rmsd=rmsd_vals.mean())


class MeanTFD(AnalysisMetric):
    """A model storing the mean TFD over a torsion profile."""

    id: int
    mean_tfd: float

    @classmethod
    def from_data(
        cls,
        torsion_id: int,
        molecule: "Molecule",
        qm_points: dict[float, Array],
        mm_points: dict[float, Array],
    ) -> Self:
        """Create a MeanTFD object by calculating the TFD between QM and MM points."""
        tfd_vals = numpy.array([get_tfd(molecule, qm_points[key], mm_points[key]) for key in qm_points])
        return cls(id=torsion_id, mean_tfd=tfd_vals.mean())


class MeanTFDCollection(AnalysisMetricCollection[MeanTFD]):
    """A collection of MeanTFD models."""

    item_type = MeanTFD


class RMSTFD(AnalysisMetric):
    """A model storing the RMS TFD over a torsion profile."""

    id: int
    rms_tfd: float

    @classmethod
    def from_data(
        cls,
        torsion_id: int,
        molecule: "Molecule",
        qm_points: dict[float, Array],
        mm_points: dict[float, Array],
    ) -> Self:
        """Create an RMSTFD object by calculating the TFD between QM and MM points."""
        tfd_vals = numpy.array([get_tfd(molecule, qm_points[key], mm_points[key]) for key in qm_points])
        return cls(
            id=torsion_id,
            rms_tfd=numpy.sqrt((tfd_vals**2).mean()),
        )


class RMSTFDCollection(AnalysisMetricCollection[RMSTFD]):
    """A collection of RMSTFD models."""

    item_type = RMSTFD


class MeanRMSDCollection(AnalysisMetricCollection[MeanRMSD]):
    """A collection of RMSD models."""

    item_type = MeanRMSD


class RMSE(AnalysisMetric):
    """A model storing the RMSE error over a torsion profile."""

    id: int
    rmse: float

    @classmethod
    def from_data(cls, torsion_id: int, qm_energies: Array, mm_energies: Array) -> Self:
        """Create an RMSE object by calculating the RMSE between QM and MM energies."""
        return cls(
            id=torsion_id,
            rmse=numpy.sqrt(((qm_energies - mm_energies) ** 2).mean()),
        )


class RMSECollection(AnalysisMetricCollection[RMSE]):
    """A collection of RMSE models."""

    item_type = RMSE


class MeanAbsoluteError(AnalysisMetric):
    """A model storing the mean absolute error over a torsion profile."""

    id: int
    mean_absolute_error: float

    @classmethod
    def from_data(cls, torsion_id: int, qm_energies: Array, mm_energies: Array) -> Self:
        """Create a MeanAbsoluteError object by calculating the mean absolute MM - QM energy."""
        return cls(
            id=torsion_id,
            mean_absolute_error=numpy.mean(numpy.abs(mm_energies - qm_energies)),
        )


class MeanAbsoluteErrorCollection(AnalysisMetricCollection[MeanAbsoluteError]):
    """A collection of MeanAbsoluteError models."""

    item_type = MeanAbsoluteError


class MeanError(AnalysisMetric):
    """A model storing the mean error over a torsion profile."""

    id: int
    mean_error: float

    @classmethod
    def from_data(cls, torsion_id: int, qm_energies: Array, mm_energies: Array) -> Self:
        """Create a MeanError object by calculating the mean MM - QM energy."""
        return cls(
            id=torsion_id,
            mean_error=numpy.mean(mm_energies - qm_energies),
        )


class MeanErrorCollection(AnalysisMetricCollection[MeanError]):
    """A collection of MeanError models."""

    item_type = MeanError


class AbsoluteBarrierHeightError(AnalysisMetric):
    """A model storing the barrier height error."""

    id: int
    absolute_barrier_height_error: float

    @classmethod
    def from_data(cls, torsion_id: int, qm_energies: Array, mm_energies: Array) -> Self:
        """Create an AbsoluteBarrierHeightError from the absolute barrier height error between QM and MM energies."""
        barrier_height_qm = qm_energies.max() - qm_energies.min()
        barrier_height_mm = mm_energies.max() - mm_energies.min()
        return cls(
            id=torsion_id,
            absolute_barrier_height_error=abs(barrier_height_mm - barrier_height_qm),
        )


class AbsoluteBarrierHeightErrorCollection(AnalysisMetricCollection[AbsoluteBarrierHeightError]):
    """A collection of AbsoluteBarrierHeightError models."""

    item_type = AbsoluteBarrierHeightError


class JSDistance(AnalysisMetric):
    """A model storing the Jensen-Shannon distances."""

    id: int
    js_distance: float
    js_temperature: float

    @classmethod
    def from_data(
        cls,
        torsion_id: int,
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
            id=torsion_id,
            # Use base 2 so that the upper limit is 1
            js_distance=jensenshannon(p_qm, p_mm, base=2),
            js_temperature=temperature,
        )


class JSDistanceCollection(AnalysisMetricCollection[JSDistance]):
    """A collection of JSDistance models."""

    item_type = JSDistance
