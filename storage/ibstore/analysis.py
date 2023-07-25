from ibstore._base.base import ImmutableModel


class DDE(ImmutableModel):
    qcarchive_id: str
    difference: float
