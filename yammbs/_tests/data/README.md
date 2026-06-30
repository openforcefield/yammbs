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

# `36966574-qm.sdf` and `36966574-mm.sdf`

Taken from a YAMMBS-dataset-submission (re-)run of Sage 2.1.0. The QCArchive ID of the molecule is 36966574. The QM molecule comes from QCArchive and the MM molecule is optimized (with YAMMBS) using Sage 2.1.0. This molecule was notable because of poor phosphate geometry. For more context see
* [YDS run](https://github.com/openforcefield/yammbs-dataset-submission/issues/68)
* [Files from YDS run](https://zenodo.org/records/17404618)
* [Bug reported in Issue #174](https://github.com/openforcefield/yammbs/issues/174)
