# Centrack

A command line interface to score centrioles in cells.

## Overview

If you want to read about Centrack, head to the Introduction.
If you want to process data head to Routine use.
If you want to install centrack, head to Set up the environment for centrack

## Introduction

Centrack is a command line interface that allows the batch processing of ome tiff files in order to automate the
centriole detection and the associated workflows.
Specifically, it orchestrates :

- the z-max projection of the raw files,
- the detection of centrioles
- the detection of the nuclei
- the assignment of the centrioles to the nearest nucleus.
- the brightness of the detected centrioles

### Repository / Source code

Eventually, centrack will be living on PyPI and can be installed using `pip install centrack`.
However, currently, this method cannot be used as centrack is not publicly available.
Therefore, it needs to be installed from the private GitHub repository (UPGON/centrack).

However, because the dependency spotipy cannot be downloaded via pip normally, one needs to download the whole
repository and then git clone the spotipy repository under src/ next to centrack.

This situation is temporary and in the near future spotipy will become a
simple dependency of centrack.
Next, when both centrack and spotipy will be publicly available, centrack will be downloadable directly from PyPI.

## Installation

! To isolate centrack from other projects, only run `pip install centrack`
within a virtual environment.

1. Install python via pyenv
2. Install poetry, system-wide with `pip install poetry`

3. Create a virtual environment with:

```shell
python3 -m venv venv-centrack
source venv-centrack/bin/activate
```

Your prompt should now be prepended with `(venv-centrack)`.

Check that you're at the correct location (simple and recommended location
is `cd ~`, i.e., your home folder).

4. Download `centrack` with:

```shell
git clone git@github.com:UPGON/centrack.git
cd centrack
```

5. As of now, you need to git clone the spotipy package in place in centrack/src/:
   !!! You need to have access to this private repo; contact Leo for setting up the permission.

```shell
cd src
git clone git@github.com:maweigert/_spotipy 
```

6. Add the programs `squash` and `score` to the PATH so that they can be run from
   the command line, without the need to type the whole path.

```shell
poetry install
```

7. Check that `centrack`'s programs are correctly installed:

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

## Procedure

1. Group all the raw OME.TIFF files into one folder called `raw`. This helps keep the structure of the processed images
   clean.
2. Run `squash` with the argument of the path to the project folder. After running the `squash`, a folder
   called `projections` is created and contains the 4-d tiff files.

```shell
squash path/to/dataset
```

3. Run `score` with the arguments source and the index of the nuclei channel (usually 0 or 3).

```shell
score path/to/dataset 0
```

4. Check that the predictions are satisfactory by looking at the folder `outlines` and at the results/scores.csv.

### Squashing the stacks to projections

`squash` expects one argument: a path to a dataset folder that contains a single folder
called `raw/`. Inside raw, you have put all the folders that contains ome.tif
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

That being said, projections files need to be converted into 8bit png files,
prior to uploading onto Labelbox platform.

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

`centrack` assumes a fixed folder structure.
Especially, the OME.tif files should be located under raw/

```text
<project_name>/
├── conditions.toml
├── projections/
├── raw/
└── scores/
```

## Training the model

- Declare the train/test split which will be used to define which images are never used as training instances (
  scripts/train_test.py).
- 
