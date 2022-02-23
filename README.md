# Centrack

A command line interface to score centrioles in cells.

## Installation

centrack is being developed with Python 3.9.5. If you don't have this version, please switch to it for instance with pyenv.

First create a virtual environment with:
```shell
$ python3 -m venv centrack-venv
$ source centrack-venv/bin/activate
```
Your prompt should now be prepended with `(centrack-venv)`.

Install `centrack` with:
```shell
$ pip install centrack
```

Check that centrack is correctly installed with:

```shell
$ squash --help
```

## Requirements
`centrack` assumes a fixed folder structure:

```text
<project_name>/
├── conditions.toml
├── projections/
├── raw/
└── scores/
```
Especially, the OME.tif files should live under raw/


## Usage

1. Navigate to the dataset folder.
2. Group all the raw OME.TIFF files into a folder called `raw`.
This helps keeping the structure of the processed images clean.
3. centrack runs on z-max projections so before running the main program `centrack score`, you may want to
run `centrack project` to save computing time in the future.
4. After running the `centrack project`, a folder called `projections` is created and contains the 4-d tiff files.
5. Now, you can swiftly run `centrack score` on the projections files.
6. You now can specify a few options to guide the scoring.
