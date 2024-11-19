import logging

import pandas

from yammbs._base.base import ImmutableModel

LOGGER = logging.getLogger(__name__)


def _normalize(qm: dict[float, float], mm: dict[float, float]) -> tuple[dict[float, float], dict[float, float]]:
    """Normalize, after sorting, a pair of QM and MM profiles to the values at the QM minimum."""
    if len(mm) == 0:
        LOGGER.warning("no mm data")
        return dict(), dict()

    _qm = dict(sorted(qm.items()))
    qm_minimum_index = min(_qm, key=_qm.get)  # type: ignore[arg-type]

    return {key: _qm[key] - _qm[qm_minimum_index] for key in _qm}, {key: mm[key] - mm[qm_minimum_index] for key in mm}


class LogSSE(ImmutableModel):
    id: int
    value: float


class LogSSECollection(list[LogSSE]):
    def to_dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame(
            [log_sse.value for log_sse in self],
            index=pandas.Index([log_sse.id for log_sse in self]),
            columns=["value"],
        )

    def to_csv(self, path: str):
        self.to_dataframe().to_csv(path)
