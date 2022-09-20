# cenfind

A command line interface to score centrioles in cells.

## Overview

If you want to read about cenfind, head to the Introduction.
If you want to process projection head to Routine use.
If you want to install cenfind, head to Set up the environment for cenfind

## Introduction

cenfind is a command line interface to detect centrioles in immunofluorescence images of human cells.
Specifically, it orchestrates :

- the z-max projection of the raw files,
- the detection of centrioles
- the detection of the nuclei
- the assignment of the centrioles to the nearest nucleus.
- the brightness of the detected centrioles

### Repository / Source code

Eventually, cenfind will be living on PyPI and will be installed using `pip install cenfind`.
However, currently, this method cannot be used as cenfind is not publicly available.
Therefore, it needs to be installed from the private GitHub repository (UPGON/cenfind).

cenfind relies on the private dependency spotipy, which cannot be downloaded via pip normally repository. Thus, follow
the instructions below.

This situation is temporary and in the near future spotipy will become a
simple dependency of cenfind.
Next, when both cenfind and spotipy will be publicly available, cenfind will be downloadable directly from PyPI.

## Installation

! To isolate cenfind from other projects, only run `pip install cenfind`
within a virtual environment.

1. Install python via pyenv
2. Install poetry, system-wide with `pip install poetry`

3. Create a virtual environment with:

```shell
python3 -m venv venv-cenfind
source venv-cenfind/bin/activate
```

Your prompt should now be prepended with `(venv-cenfind)`.

Check that you're at the correct location (simple and recommended location
is `cd ~`, i.e., your home folder).

4. Download `cenfind` with:

```shell
git clone git@github.com:UPGON/cenfind.git
cd cenfind
```

5. As of now, you need to git clone the spotipy package in place in cenfind/src/:
   !!! You need to have access to this private repo; contact Leo for setting up the permission.

```shell
cd ..
git clone git@github.com:maweigert/spotipy.git
pip install -e spotipy/
```

6. Add the programs `squash` and `score` to the PATH so that they can be run from
   the command line, without the need to type the whole path.

```shell
cd cenfind
poetry install
```

7. Check that `cenfind`'s programs are correctly installed:

```shell
squash --help
```

Note: it may take a few seconds.

8. In case of updates, get the last version:

```shell
git pull
poetry install
```

A common session involves running `squash` then `score`. Below, we
describe each program, their input, the algorithm and the expected output.

## API

cenfind consists of the `Dataset` and the `Field` classes.

A Dataset represents a collection of related fields, i.e., same pixel size, same channels, same cell type.

It should:

- return the name
- iterate over the fields,
- construct the file name for the projections and the z-stacks
- read the fields.txt
- write the fields.txt file
- set up the folders projections, predictions, visualisations and statistics
- set and get the splits

A Field represents a field of view and should:

- construct file names for projections, annotation
- get Dataset
- load the projection as np.ndarray
- load the channel as np.ndarray
- load annotation as np.ndarray
- load mask as np.ndarray

Using those two objects, cenfind should

- detect centrioles (data, model) => points,
- extract nuclei (data, model) => contours,
- assign centrioles to nuclei (contours, points) => pairs
- outline centrioles and nuclei (data, points) => image
- create composite vignettes (data) => composite_image
- flag partial nuclei (contours, tolerance) => contours
- compare predictions with annotation (points, points) => metrics_namespace

## Routines

1. Group all the raw OME.TIFF files into one folder called `raw`. This helps keep the structure of the processed images
   clean.
2. Run `squash` with the argument of the path to the project folder and the suffix of the raw files. After running
   the `squash`, a folder
   called `projections` is created and contains the 4-d tiff files.

```shell
squash path/to/ds .ome.tif
```

3. Run `score` with the arguments source and the index of the nuclei channel (usually 0 or 3).

```shell
score path/to/ds 0
```

4. Check that the predictions are satisfactory by looking at the folder `outlines` and at the results/scores.csv.

### Squashing the stacks to projections

`squash` expects two argument: a path to a dataset folder that contains a single folder
called `raw/` and the type of raw images (.ome.tif, .stk). Inside raw, you have put all the folders that contains
ome.tif
files. These ome.tif files are fetched, squashed and saved to `projections/`, next to the `raw/` folder.

The files are loaded using tifffile into the memory (intensive; as each file may
be 4.2 GB in size). Each file as up to 5 dimensions (TCZYX) but so far only
CZYX are supported by squash. The first step is to figure out the position
of the Z-axis. Once this has been determined, the array is max-projected
along the z-axis and the file is saved under projections/filename_max.tif,
where filename is extracted from the original filename. This operation is
repeated for each file in raw.

No further preprocessing is applied to the projections, for instance the bit
depth is unchanged (16bit) and no contrast adjustment is applied.

However, projections files need to be converted into 8bit png files,
prior to uploading onto Labelbox platform. This conversion is further explained in the tutorial "Experimenting and
Extending cenfind with new datasets"

### Scoring the centrioles

The neural network SPOTNET is run on each centriolar channel and returns a list of the coordinates of the detected
centrioles. The coordinates are represented as (row, col) with respect to the image dimensions.
In parallel, the cells are located by segmenting the nuclei from the DAPI channel. Once the nuclei are segmented. Each
centriole is assigned to the nearest nucleus but at most 50 px away.

### Saving the predictions

When centrioles and nuclei are detected automatically, the results are called predictions. When the predictions are
checked by an experimenter, they become annotations.
Two prediction types are saved: the coordinates of the centrioles and the mask containing the nuclei, in which each
nucleus is labelled with an index, while the background is set to 0.

## Requirements

`cenfind` assumes a fixed folder structure.
Especially, the OME.tif files should be located under raw/

```text
<project_name>/
├── projections/
├── raw/
└── scores/
```

## Training the model

- Declare the train/test split which will be used to define which images are never used as training instances (
  scripts/train_test.py).
- TODO
