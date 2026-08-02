# Building the Docs

Although documentation for YAMMBS is [readily available online], it is sometimes useful to build a local version, such as when:

- developing new pages which you wish to preview without having to wait
  for ReadTheDocs to finish building.
- debugging errors which occur when building on ReadTheDocs.

In these cases, the docs can be built locally by doing the following:

```bash
git clone https://github.com/openforcefield/yammbs.git
cd yammbs
pixi install -e docs
rm -rf docs/api/generated docs/_build/html
pixi run -e docs build_docs
```

The above will yield a new directory named `docs/_build/html` which will
contain the built HTML files which can be viewed in your local browser.

[readily available online]: https://yammbs.readthedocs.io/en/latest/
