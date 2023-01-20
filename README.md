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
cenfind squash --help
```

## Basic usage
Before scoring the cells, you need to prepare the dataset folder. 
`cenfind` assumes a fixed folder structure. 
In the following we will assume that the .ome.tif files are all immediately in raw/. 
Each file is a z-stack field of view (referred to as field, in the following) containing 4 channels (0, 1, 2, 3). The channel 0 contains the nuclei and the channels 1-3 contains centriolar markers.

```text
<project_name>/
└── raw/
```
2. Run `prepare` to initialise the folder with a list of channels and output folders:
```shell
cenfind prepare /path/to/dataset <list channels of centrioles, like 1 2 3, (if 0 is the nucleus channel)>
```

2. Run `squash` with the path to the project folder and the suffix of the raw files. `projections/` is populated with the max-projections `*_max.tif` files.
```shell
cenfind squash path/to/dataset
```

3. Run `score` with the arguments source, the index of the nuclei channel (usually 0 or 3), the channel to score and the path to the model. You need to download it from https://figshare.com/articles/software/Cenfind_model_weights/21724421
```shell
cenfind score /path/to/dataset /path/to/model/ 0 1 2 3 --projection_suffix '_max'
```

4. Check that the predictions are satisfactory by looking at the folders `visualisation/` and `statistics/`

5. If you interested in categorising the number of centrioles, run `cenfind analyse path/to/dataset --by <well>` the --by option is interesting if you want to group your scoring by well, if the file names obey to the rule `<WELLID_FOVID>`.

## Running `cenfind score` in the background

When you exit the shell, running programs receive the SIGHUP, which aborts them. This is undesirable if you need to close your shell for some reasons. Fortunately, you can make your program ignore this signal by prepending the program with the `nohup` command. Moreover, if you want to run your program in the background, you can append the ampersand `&`. In practice, run `nohup cenfind score ... &` instead of `cenfind score ...`.

The output will be written to the file `nohup.out` and you can peek the progress by running `tail -F nohup.out`, the flag `-F` will refresh the screen as the file is being written. Enter Ctrl-C to exit the tail program.

If you want to kill the program score, run  `jobs` and then run `kill <jobid>`. If you see no jobs, check the log `nohup.out`; it can be done or the program may have crashed, and you can check the error there.


## Internal API

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
