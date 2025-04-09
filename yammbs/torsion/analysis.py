import logging
from typing import TYPE_CHECKING

from yammbs._base.base import ImmutableModel

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas


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


class Minima(ImmutableModel):
    id: int
    minima: list[float]


class RMSD(ImmutableModel):
    id: int
    rmsd: float


class RMSDCollection(list[RMSD]):
    def to_dataframe(self) -> "pandas.DataFrame":
        import pandas

        return pandas.DataFrame(
            [rmsd.rmsd for rmsd in self],
            index=pandas.Index([rmsd.id for rmsd in self]),
            columns=["rmsd"],
        )

    def to_csv(self, path: str):
        self.to_dataframe().to_csv(path)


class RMSE(ImmutableModel):
    id: int
    rmse: float


class RMSECollection(list[RMSE]):
    def to_dataframe(self) -> "pandas.DataFrame":
        import pandas

        return pandas.DataFrame(
            [rmse.rmse for rmse in self],
            index=pandas.Index([rmse.id for rmse in self]),
            columns=["rmse"],
        )

    def to_csv(self, path: str):
        self.to_dataframe().to_csv(path)


class MeanError(ImmutableModel):
    id: int
    mean_error: float


class MeanErrorCollection(list[MeanError]):
    def to_dataframe(self) -> "pandas.DataFrame":
        import pandas

        return pandas.DataFrame(
            [mean_error.mean_error for mean_error in self],
            index=pandas.Index([mean_error.id for mean_error in self]),
            columns=["mean_error"],
        )

    def to_csv(self, path: str):
        self.to_dataframe().to_csv(path)

class JSDivergence(ImmutableModel):
    id: int
    js_divergence: float
    temperature: float

class JSDivergenceCollection(list[JSDivergence]):
    def to_dataframe(self) -> "pandas.DataFrame":
        import pandas

        return pandas.DataFrame(
            [(js_divergence.js_divergence, js_divergence.temperature) for js_divergence in self],
            index=pandas.Index([js_divergence.id for js_divergence in self]),
            columns=["js_divergence", "temperature"],
        )

    def to_csv(self, path: str):
        self.to_dataframe().to_csv(path)