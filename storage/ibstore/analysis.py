import pandas

from ibstore._base.base import ImmutableModel


class DDE(ImmutableModel):
    qcarchive_id: str
    difference: float


class DDECollection(list):
    def to_dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame(
            [dde.difference for dde in self],
            index=pandas.Index([dde.qcarchive_id for dde in self]),
            columns=["difference"],
        )

    def to_csv(self, path: str):
        self.to_dataframe().to_csv(path)
