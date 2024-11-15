import pandas

from yammbs._base.base import ImmutableModel


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
