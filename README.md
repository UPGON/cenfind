# cenfind

A command line interface to score cells for centrioles.

## Introduction

`cenfind` is a command line interface to detect and assign centrioles in immunofluorescence images of human cells. Specifically, it orchestrates:

- the z-max projection of the raw files;
- the detection of centrioles;
- the detection of the nuclei;
- the assignment of the centrioles to the nearest nucleus.

## Installation

1. Download `cenfind` with:

```shell
python -m venv test_cenfind
source test_cenfind/bin/activate
pip install cenfind
pip install tensorflow
pip install -e projects/spotipy
```

5. As of now, you need to install the spotipy package from the git repository https://github.com/maweigert/spotipy:

```shell
git clone git@github.com:maweigert/spotipy.git
```

6. Add manually the package spotipy
```shell
pip install -e ../spotipy/
```

7. Check that `cenfind`'s programs are correctly installed by running:

```shell
squash --help
```

## Basic usage
Before scoring the cells, you need to prepare the dataset folder. `cenfind` assumes a fixed folder structure. In the following we will assume that the .ome.tif files are all immediately in raw/. Each field of view is a z-stack containing 4 channels (0, 1, 2, 3). The channel 0 contains the nuclei and the channels 1-3 contains centriolar markers.
```text
<project_name>/
└── raw/
```
2. Run `prepare` to initialise the folder with a list of fields and output folders:
```shell
prepare /path/to/dataset 1 2 3 [--pixel_size, [--projection_suffix]]
```

2. Run `squash` with the argument of the path to the project folder and the suffix of the raw files. `projections/` is populated with the max-projections `*_max.tif` files.
```shell
squash path/to/dataset
```

3. Run `score` with the arguments source and the index of the nuclei channel (usually 0 or 3).
```shell
score path model channel_nuclei channels factor
```

4. Check that the predictions are satisfactory by looking at the folder `outlines` and at the results/scores.csv.

## API

`cenfind` consists of two core classes: `Dataset` and `Field`. The class `Dataset` represents a collection of fields with identical pixel size, channels, and cell type. The class `Field` represents a field of view and is responsible for calling the detection procedure and for managing (storing and writing) the predictions.

Using those two objects, `cenfind` should

- detect centrioles (data, model) => points,
- extract nuclei (data, model) => contours,
- assign centrioles to nuclei (contours, points) => pairs
- outline centrioles and nuclei (data, points) => image
- create composite vignettes (data) => composite_image
- flag partial nuclei (contours, tolerance) => contours
- compare predictions with annotation (points, points) => metrics_namespace
