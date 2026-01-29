"""Analysis methods for torsion profiles."""

import logging
from abc import ABC
from enum import Enum
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Generic,
    Self,
    TypeVar,
)

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


def _compute_rms(vals: numpy.ndarray) -> float:
    """Compute the root mean square of an array."""
    return float(numpy.sqrt((vals**2).mean()))


def _compute_mean(vals: numpy.ndarray) -> float:
    """Compute the mean of an array."""
    return float(vals.mean())


class MetricType(Enum):
    """Enum to distinguish between coordinate-based and energy-based metrics."""

    COORDINATE_BASED = "coordinate_based"
    ENERGY_BASED = "energy_based"


def _compute_metric_from_coordinates(
    molecule: "Molecule",
    qm_points: dict[float, Array],
    mm_points: dict[float, Array],
    metric_func,
    aggregation_func,
) -> float:
    """Compute coordinate-based metrics (RMSD/TFD) with different aggregations.

    Args:
        molecule: The molecule object.
        qm_points: QM coordinate points indexed by grid angle.
        mm_points: MM coordinate points indexed by grid angle.
        metric_func: Function to compute per-point metric (get_rmsd or get_tfd).
        aggregation_func: Function to aggregate values (_compute_rms or _compute_mean).

    Returns:
        The aggregated metric value.

    """
    vals = numpy.array([metric_func(molecule, qm_points[key], mm_points[key]) for key in qm_points])
    return aggregation_func(vals)


class AnalysisMetric(ABC, ImmutableModel):
    """A model storing a single analysis metric."""

    id: int

    metric_type: ClassVar[MetricType]


AnalysisMetricTypeVar = TypeVar("AnalysisMetricTypeVar", bound=AnalysisMetric)


class AnalysisMetricCollection(ABC, Generic[AnalysisMetricTypeVar], list[AnalysisMetricTypeVar]):
    """A generic collection class for typed lists of analysis metrics."""

    item_type: ClassVar[type[AnalysisMetric]]  # Set in subclasses to specific metric type

    @classmethod
    def get_item_type(cls) -> type[AnalysisMetric]:
        """Retrieve the type of items in the collection."""
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


# Global registry for all metric collections
_METRIC_REGISTRY: list[type[AnalysisMetricCollection]] = []


def register_metric(collection_class: type[AnalysisMetricCollection]):
    """Register a metric collection for auto-discovery."""
    _METRIC_REGISTRY.append(collection_class)
    return collection_class


def get_all_metric_collections() -> list[type[AnalysisMetricCollection]]:
    """Get all registered metric collections."""
    return _METRIC_REGISTRY.copy()


class Minima(AnalysisMetric):
    """A model storing the minima of a torsion profile."""

    metric_type: ClassVar[MetricType] = MetricType.ENERGY_BASED

    id: int
    minima: list[float]


class RMSRMSD(AnalysisMetric):
    """A model storing the RMS RMSD over a torsion profile."""

    metric_type: ClassVar[MetricType] = MetricType.COORDINATE_BASED

    id: int
    rms_rmsd: float

    @classmethod
    def from_data(
        cls,
        torsion_id: int,
        molecule: "Molecule",
        qm_points: dict[float, Array],
        mm_points: dict[float, Array],
        **kwargs,
    ) -> Self:
        """Create an RMSD object by calculating the RMSD between QM and MM points."""
        return cls(
            id=torsion_id,
            rms_rmsd=_compute_metric_from_coordinates(molecule, qm_points, mm_points, get_rmsd, _compute_rms),
        )


@register_metric
class RMSRMSDCollection(AnalysisMetricCollection[RMSRMSD]):
    """A collection of RMSD models."""

    item_type = RMSRMSD


class MeanRMSD(AnalysisMetric):
    """A model storing the mean RMSD over a torsion profile."""

    metric_type: ClassVar[MetricType] = MetricType.COORDINATE_BASED

    id: int
    mean_rmsd: float

    @classmethod
    def from_data(
        cls,
        torsion_id: int,
        molecule: "Molecule",
        qm_points: dict[float, Array],
        mm_points: dict[float, Array],
        **kwargs,
    ) -> Self:
        """Create an RMSD object by calculating the RMSD between QM and MM points."""
        return cls(
            id=torsion_id,
            mean_rmsd=_compute_metric_from_coordinates(molecule, qm_points, mm_points, get_rmsd, _compute_mean),
        )


class MeanTFD(AnalysisMetric):
    """A model storing the mean TFD over a torsion profile."""

    metric_type: ClassVar[MetricType] = MetricType.COORDINATE_BASED

    id: int
    mean_tfd: float

    @classmethod
    def from_data(
        cls,
        torsion_id: int,
        molecule: "Molecule",
        qm_points: dict[float, Array],
        mm_points: dict[float, Array],
        **kwargs,
    ) -> Self:
        """Create a MeanTFD object by calculating the TFD between QM and MM points."""
        return cls(
            id=torsion_id,
            mean_tfd=_compute_metric_from_coordinates(molecule, qm_points, mm_points, get_tfd, _compute_mean),
        )


@register_metric
class MeanTFDCollection(AnalysisMetricCollection[MeanTFD]):
    """A collection of MeanTFD models."""

    item_type = MeanTFD


class RMSTFD(AnalysisMetric):
    """A model storing the RMS TFD over a torsion profile."""

    metric_type: ClassVar[MetricType] = MetricType.COORDINATE_BASED

    id: int
    rms_tfd: float

    @classmethod
    def from_data(
        cls,
        torsion_id: int,
        molecule: "Molecule",
        qm_points: dict[float, Array],
        mm_points: dict[float, Array],
        **kwargs,
    ) -> Self:
        """Create an RMSTFD object by calculating the TFD between QM and MM points."""
        return cls(
            id=torsion_id,
            rms_tfd=_compute_metric_from_coordinates(molecule, qm_points, mm_points, get_tfd, _compute_rms),
        )


@register_metric
class RMSTFDCollection(AnalysisMetricCollection[RMSTFD]):
    """A collection of RMSTFD models."""

    item_type = RMSTFD


@register_metric
class MeanRMSDCollection(AnalysisMetricCollection[MeanRMSD]):
    """A collection of RMSD models."""

    item_type = MeanRMSD


class RMSE(AnalysisMetric):
    """A model storing the RMSE error over a torsion profile."""

    metric_type: ClassVar[MetricType] = MetricType.ENERGY_BASED

    id: int
    rmse: float

    @classmethod
    def from_data(
        cls,
        torsion_id: int,
        qm_energies: Array,
        mm_energies: Array,
    ) -> Self:
        """Create an RMSE object by calculating the RMSE between QM and MM energies."""
        return cls(
            id=torsion_id,
            rmse=numpy.sqrt(((qm_energies - mm_energies) ** 2).mean()),
        )


@register_metric
class RMSECollection(AnalysisMetricCollection[RMSE]):
    """A collection of RMSE models."""

    item_type = RMSE


class MeanAbsoluteError(AnalysisMetric):
    """A model storing the mean absolute error over a torsion profile."""

    metric_type: ClassVar[MetricType] = MetricType.ENERGY_BASED

    id: int
    mean_absolute_error: float

    @classmethod
    def from_data(
        cls,
        torsion_id: int,
        qm_energies: Array,
        mm_energies: Array,
    ) -> Self:
        """Create a MeanAbsoluteError object by calculating the mean absolute MM - QM energy."""
        return cls(
            id=torsion_id,
            mean_absolute_error=numpy.mean(numpy.abs(mm_energies - qm_energies)),
        )


@register_metric
class MeanAbsoluteErrorCollection(AnalysisMetricCollection[MeanAbsoluteError]):
    """A collection of MeanAbsoluteError models."""

    item_type = MeanAbsoluteError


class MeanError(AnalysisMetric):
    """A model storing the mean error over a torsion profile."""

    metric_type: ClassVar[MetricType] = MetricType.ENERGY_BASED

    id: int
    mean_error: float

    @classmethod
    def from_data(
        cls,
        torsion_id: int,
        qm_energies: Array,
        mm_energies: Array,
    ) -> Self:
        """Create a MeanError object by calculating the mean MM - QM energy."""
        return cls(
            id=torsion_id,
            mean_error=numpy.mean(mm_energies - qm_energies),
        )


@register_metric
class MeanErrorCollection(AnalysisMetricCollection[MeanError]):
    """A collection of MeanError models."""

    item_type = MeanError


class AbsoluteBarrierHeightError(AnalysisMetric):
    """A model storing the barrier height error."""

    metric_type: ClassVar[MetricType] = MetricType.ENERGY_BASED

    id: int
    absolute_barrier_height_error: float

    @classmethod
    def from_data(
        cls,
        torsion_id: int,
        qm_energies: Array,
        mm_energies: Array,
    ) -> Self:
        """Create an AbsoluteBarrierHeightError from the absolute barrier height error between QM and MM energies."""
        barrier_height_qm = qm_energies.max() - qm_energies.min()
        barrier_height_mm = mm_energies.max() - mm_energies.min()
        return cls(
            id=torsion_id,
            absolute_barrier_height_error=abs(barrier_height_mm - barrier_height_qm),
        )


@register_metric
class AbsoluteBarrierHeightErrorCollection(AnalysisMetricCollection[AbsoluteBarrierHeightError]):
    """A collection of AbsoluteBarrierHeightError models."""

    item_type = AbsoluteBarrierHeightError


class JSDistance(AnalysisMetric):
    """A model storing the Jensen-Shannon distances."""

    metric_type: ClassVar[MetricType] = MetricType.ENERGY_BASED

    id: int
    js_distance: float
    js_temperature: float

    @classmethod
    def from_data(
        cls,
        torsion_id: int,
        qm_energies: Array,
        mm_energies: Array,
        temperature: float = 500.0,  # Kelvin
    ) -> Self:
        """Create a JSDistance object from supplied data.

        Do this by calculating the Jensen-Shannon distance
        between the two distributions generated with Boltzmann inversion at the
        specified temperature, using base 2 logs. A distance of 0 indicates
        identical distributions, while a distance of 1 indicates completely
        non-overlapping distributions.

        Args:
            torsion_id: ID of the torsion.
            qm_energies: QM energies array.
            mm_energies: MM energies array.
            temperature: Temperature in Kelvin for Boltzmann inversion.

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


@register_metric
class JSDistanceCollection(AnalysisMetricCollection[JSDistance]):
    """A collection of JSDistance models."""

    item_type = JSDistance
