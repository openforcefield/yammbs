(installation)=

# Installation

:::{note}
YAMMBS is currently developed for internal use at Open Force Field and is not available via conda-forge or PyPI.
It must be installed from source via a git clone.
:::

(installation/source)=

## Installing from source

YAMMBS requires cloning the repository and installing from source. The recommended installation method uses [Mamba](https://github.com/mamba-org/mamba) or [Conda](https://docs.conda.io/en/latest/) to manage dependencies:

```shell-session
$ git clone https://github.com/openforcefield/yammbs.git
$ cd yammbs
$ mamba env create -n yammbs -f devtools/conda-envs/dev.yaml
$ mamba activate yammbs
$ pip install -e .
```

This will:
1. Clone the YAMMBS repository
2. Create a new conda environment with all dependencies
3. Install YAMMBS in editable mode

If you do not have Mamba or Conda installed, see the [ecosystem installation documentation].

[ecosystem installation documentation]: inv:openff.docs#install

(installation/platforms)=

## OS support

YAMMBS is pure Python, and we expect it to work on any platform that supports its dependencies.
Our automated testing takes place on both (x86) MacOS and Ubuntu Linux.

(installation/checkmol)=

## Checkmol dependency

YAMMBS uses [checkmol](https://homepage.univie.ac.at/norbert.haider/cheminf/cmmm.html) for molecular structure analysis. On most systems, you will need to install it separately:

### macOS

```shell-session
$ brew install fpc
$ curl https://homepage.univie.ac.at/norbert.haider/download/chemistry/checkmol/checkmol.pas > checkmol.pas
$ fpc checkmol.pas -S2
$ sudo mv checkmol /usr/local/bin/
```

### Ubuntu/Debian

```shell-session
$ sudo apt-get install fp-compiler
$ curl https://homepage.univie.ac.at/norbert.haider/download/chemistry/checkmol/checkmol.pas > checkmol.pas
$ fpc checkmol.pas -S2
$ sudo mv checkmol /usr/local/bin/
```
