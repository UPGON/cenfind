# cenfind

A command line interface to score cells for centrioles.

## Introduction

`cenfind` is a command line interface to detect and assign centrioles in immunofluorescence images of human cells. Specifically, it orchestrates:

- the z-max projection of the raw files;
- the detection of centrioles;
- the detection of the nuclei;
- the assignment of the centrioles to the nearest nucleus.

## Installation
1. Install python via pyenv
2. Download and set up 3.9.5 as local version
3. Install poetry, system-wide with `pip install poetry`

Check that you're at the correct location (a simple and recommended location
is `cd ~`, i.e., your home folder).

4. Download `cenfind` with:

```shell
git clone git@github.com:UPGON/cenfind.git
git clone git@github.com:maweigert/spotipy.git
```

5. As of now, you need to install the spotipy package from the git repository https://github.com/maweigert/spotipy:
   !!! You need to have access to this private repo; contact Leo for setting up the permission.

```shell
cd cenfind
```
6. Activate the virtual environment using poetry
```shell
poetry shell
```
Your prompt should now be prepended with `(cenfind-py3.9)`.

Note: if your python version is not supported, install the one recommended with pyenv, the set it up and run `poetry env use $(which python)`. Then, repeat the step. 

6. Add the programs `squash` and `score` to the PATH with the following commands, so that they can be run from the command line, without the need to type the whole path.

```shell
poetry install
```

6. Add manually the package spotipy
```shell
pip install -e ../spotipy/
```

7. Check that `cenfind`'s programs are correctly installed by running:

```shell
squash --help
```

8. In case of updates, get the last version:

```shell
git pull
poetry install
```

## Basic usage
Before scoring the cells, you need to prepare the dataset folder. `cenfind` assumes a fixed folder structure. In the following we will assume that the .ome.tif files are all immediately in raw/. Each field of view is a z-stack containing 4 channels (0, 1, 2, 3). The channel 0 contains the nuclei and the channels 1-3 contains centriolar markers.
```text
<project_name>/
└── raw/
```
2. Run `setup` to initialise the folder with a list of fields and output folders:
```shell
prepare /path/to/dataset <list channels of centrioles, like 1 2 3, (0 should be the nucleus channel)>
```

2. Run `squash` with the argument of the path to the project folder and the suffix of the raw files. `projections/` is populated with the max-projections `*_max.tif` files.
```shell
squash path/to/ds .ome.tif
```

3. Run `score` with the arguments source and the index of the nuclei channel (usually 0 or 3).
```shell
score /path/to/dataset ./model/master/ 0 1 2 3 --projection_suffix '_max'
```

4. Check that the predictions are satisfactory by looking at the folder `outlines` and at the results/scores.csv.

## API

`cenfind` consists of two core classes: `Dataset` and `Field`.

A `Dataset` represents a collection of related fields, i.e., same pixel size, same channels, same cell type.

It should:
- return the name
- iterate over the fields,
- construct the file name for the projections and the z-stacks
- read the fields.txt
- write the fields.txt file
- set up the folders projections, predictions, visualisations and statistics
- set and get the splits

A `Field` represents a field of view and should:

- construct file names for projections, annotation
- get Dataset
- load the projection as np.ndarray
- load the channel as np.ndarray
- load annotation as np.ndarray
- load mask as np.ndarray

Using those two objects, `cenfind` should

- detect centrioles (data, model) => points,
- extract nuclei (data, model) => contours,
- assign centrioles to nuclei (contours, points) => pairs
- outline centrioles and nuclei (data, points) => image
- create composite vignettes (data) => composite_image
- flag partial nuclei (contours, tolerance) => contours
- compare predictions with annotation (points, points) => metrics_namespace
