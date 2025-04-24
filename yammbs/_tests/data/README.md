# tiny-opt.json

This file is built by calling `make_tiny_opt.py` like this:

``` shell
curl 'https://raw.githubusercontent.com/openforcefield/sage-2.1.0/main/inputs-and-outputs/data-sets/opt-set-for-fitting-2.1.0.json' |
	jq '{
			entries: {
					 "https://api.qcarchive.molssi.org:443/": .entries["https://api.qcarchive.molssi.org:443/"][:200]
					 },
			provenance: .provenance,
			type: .type
		  }' |
	python make_tiny_opt.py > tiny-opt.json
```

# yammbs/torsiondrive-data.json

```
In [1]: from yammbs.torsion.inputs import QCArchiveTorsionDataset

In [2]: from openff.qcsubmit.results import TorsionDriveResultCollection

In [3]: x = TorsionDriveResultCollection.parse_file("yammbs/_tests/data/qcsubmit/filtered-supp-td.json")

In [4]: y = QCArchiveTorsionDataset.from_qcsubmit_collection(x)

In [5]: with open('y.json', 'w') as f:
   ...:     f.write(y.json())
   ...:
<ipython-input-5-4e2b34935460>:2: PydanticDeprecatedSince20: The `json` method is deprecated; use `model_dump_json` instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.11/migration/
  f.write(y.json())

In [6]: !cp y.json yammbs/_tests/data/yammbs/torsiondrive-data.json
```
