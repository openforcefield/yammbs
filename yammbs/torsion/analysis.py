import logging
from typing import TYPE_CHECKING

from yammbs._base.base import ImmutableModel

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas


def _normalize(qm: dict[float, float], mm: dict[float, float]) -> tuple[dict[float, float], dict[float, float]]:
    """Normalize, after sorting, a pair of QM and MM profiles to the values at the QM minimum."""
    if len(mm) == 0:
        LOGGER.warning("no mm data")
        return dict(), dict()

    _qm = dict(sorted(qm.items()))
    qm_minimum_index = min(_qm, key=_qm.get)  # type: ignore[arg-type]

    return {key: _qm[key] - _qm[qm_minimum_index] for key in _qm}, {
        key: mm[key] - mm[qm_minimum_index] for key in sorted(mm)
    }


class LogSSE(ImmutableModel):
    id: int
    log_sse: float


class LogSSECollection(list[LogSSE]):
    def to_dataframe(self) -> "pandas.DataFrame":
        import pandas

        return pandas.DataFrame(
            [log_sse.log_sse for log_sse in self],
            index=pandas.Index([log_sse.id for log_sse in self]),
            columns=["log_sse"],
        )

    def to_csv(self, path: str):
        self.to_dataframe().to_csv(path)


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
            index=pandas.Index([rmsd.qcarchive_id for rmsd in self]),
            columns=["rmsd"],
        )

    def to_csv(self, path: str):
        self.to_dataframe().to_csv(path)
