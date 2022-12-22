# CenFind

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
3. Set up Python interpreter
```shell
pyenv local 3.9.5
pyenv global 3.9.5
```
4. Create a virtual environment for CenFind
```shell
python -m venv venv-cenfind
source venv-cenfind/bin/activate
```

5. Check that `cenfind`'s programs are correctly installed by running:

```shell
squash --help
```

## Basic usage
Before scoring the cells, you need to prepare the dataset folder. 
`cenfind` assumes a fixed folder structure. 
In the following we will assume that the .ome.tif files are all immediately in raw/. 
Each field of view is a z-stack containing 4 channels (0, 1, 2, 3). The channel 0 contains the nuclei and the channels 1-3 contains centriolar markers.

```text
<project_name>/
└── raw/
```
2. Run `prepare` to initialise the folder with a list of fields and output folders:
```shell
prepare /path/to/dataset <list channels of centrioles, like 1 2 3, (if 0 is the nucleus channel)>
```

2. Run `squash` with the argument of the path to the project folder and the suffix of the raw files. `projections/` is populated with the max-projections `*_max.tif` files.
```shell
squash path/to/dataset
```

3. Run `score` with the arguments source, the index of the nuclei channel (usually 0 or 3), the channel to score and the path to the model. You need to download it from https://figshare.com/articles/software/Cenfind_model_weights/21724421
```shell
score /path/to/dataset /path/to/model/ 0 1 2 3 --projection_suffix '_max'
```

For reference:
```shell
score -h
CENFIND: Automatic centriole scoring

positional arguments:
  path                  Path to the dataset
  model                 Absolute path to the model folder
  channel_nuclei        Channel index for nuclei segmentation, e.g., 0 or 3
  channels              Channel indices to analyse, e.g., 1 2 3

optional arguments:
  -h, --help            show this help message and exit
  --vicinity VICINITY   Distance threshold in micrometer (default: -5 um)
  --factor FACTOR       Factor to use: given a 2048x2048 image, 256 if 63x; 2048 if 20x:
  --projection_suffix PROJECTION_SUFFIX
                        Projection suffix (`_max` (default) or `_Projected`
```

4. Check that the predictions are satisfactory by looking at the folders `visualisation/` and `statistics/`

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
