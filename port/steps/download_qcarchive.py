import importlib
from typing import Iterable

from qcportal import FractalClient


def download_qcarchive(
    collection_type: str,
    data_set: str,
    cache_file: str,
    filter_smiles: Iterable[str] = tuple(),
    spec_name: str = "default",
):
    print("starting client")
    client = FractalClient()

    print("importing collection class")
    collection_class = getattr(
        importlib.import_module("openff.qcsubmit.results"),
        collection_type,
    )

    print("retrieving data")
    data = collection_class.from_server(
        client=client,
        datasets=data_set,
        spec_name="default",
    )

    print("filtering data")
    ind_to_del = []
    for i, item in enumerate(data.entries["https://api.qcarchive.molssi.org:443/"]):
        for smiles in filter_smiles:
            if smiles.upper() in item.cmiles or smiles.lower() in item.cmiles:
                ind_to_del.append(i)
                continue

    for ind in sorted(ind_to_del, reverse=True):
        print("deleting implicit hydrogen entry: ", ind)
        del data.entries["https://api.qcarchive.molssi.org:443/"][ind]

    print("writing data")
    with open(cache_file, "w") as _file:
        _file.write(data.json())
