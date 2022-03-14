# Centrack

A command line interface to score centrioles in cells.

## Installation

1. Install pyenv and build python 3.9.5; set it as the local and the global version using
```shell
pyenv install 3.9.5
pyenv local 3.9.5
pyenv global 3.9.5
```

3. Create a virtual environment with:
```shell
python3 -m venv centrack-venv
source centrack-venv/bin/activate
```
Your prompt should now be prepended with `(centrack-venv)`.

4. Install `centrack` with:
```shell
git clone git@github.com:UPGON/centrack.git
cd centrack
```

5. Check that `centrack` is correctly installed with:

```shell
$ (centrack-venv) squash --help
```

6. In case of updates, get the last version:
```shell
git pull
poetry install

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
Especially, the OME.tif files should be located under raw/


## Usage

1. Navigate to the dataset folder.
2. Group all the raw OME.TIFF files into one folder called `raw`. This helps keep the structure of the processed images clean.
3. Run `squash` with the argument of the path to the project folder. After running the `squash`, a folder called `projections` is created and contains the 4-d tiff files.
```shell
project path/to/dataset --format garcia
```
4. Run `score` with the arguments source and the channel to use.
```shell
score path/to/dataset --channel <markername>
```
5. Check that the predictions are satisfactory by looking at the folder `outlines` and at the results/scores.csv.
6. Run `reduce` to count the number of centrioles per cell. 
