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
