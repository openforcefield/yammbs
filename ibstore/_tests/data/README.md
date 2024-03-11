# tiny-opt.json

This file is built by calling this code

``` python
import logging

import click
from openff.qcsubmit.results import OptimizationResultCollection

from ibstore.cached_result import CachedResultCollection

logging.getLogger("openff").setLevel(logging.ERROR)


@click.command()
@click.option("--dataset", "-d")
@click.option("--output", "-o")
def main(dataset, output):
    print("loading collection")
    opt = OptimizationResultCollection.parse_file(dataset)
    print("caching records")
    crc = CachedResultCollection.from_qcsubmit_collection(opt)

    print(f"serializing to json in {output}")
    with open(output, "w") as out:
        out.write(crc.to_json(indent=2))


if __name__ == "__main__":
    main()
```

with the arguments `-d tiny-opt-dataset.json -o tiny-opt.json`.
`tiny-opt-dataset.json` consists of the first 200 records from the [Sage
2.1.0][sage] optimization dataset.

[sage]: https://raw.githubusercontent.com/openforcefield/sage-2.1.0/main/inputs-and-outputs/data-sets/opt-set-for-fitting-2.1.0.json
