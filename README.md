# CenFind - A command line interface to score cells for centrioles.

## Introduction

`cenfind` is a command line interface to detect and assign centrioles in immunofluorescence images of human cells. Specifically, it orchestrates:

- the z-max projection of the raw files;
- the detection of centrioles;
- the detection of the nuclei;
- the assignment of the centrioles to the nearest nucleus.

## Installing CenFind

1. Install python via pyenv

2. Download and set up Pyton 3.9.5 as local version:
```shell
pyenv install 3.9.5
pyenv local 3.9.5
pyenv global 3.9.5
```

3. Create a virtual environment for CenFind
```shell
python -m venv venv-cenfind
source venv-cenfind/bin/activate
```

4. Install `cenfind` with `pip`:
```shell
pip install cenfind
```

6. Check that `cenfind` is correctly installed by running:
```shell
cenfind --help
```

## Prelude

Cenfind assumes a fixed folder structure. 
We thus provide the subprogram `prepare` to set up the folder structure. 
Stacks or projections must be located in raw/ or projections respectively. 
Note that if you have no projections, the subprogram will need to write the projections before it can score the fields of view (FOVs). 

Run `prepare` to initialise the folder with a list of channels and output folders:
```shell
cenfind prepare /path/to/dataset --projection_suffix '_max'
```
If no projection suffix is provided, projection names are assumed to have no suffix.
Otherwise you need to specify it:
```shell
cenfind prepare /path/to/dataset --projection_suffix '_max'
```

If you have to label the dataset for future training, specify, using the `--splits` option, the channels to be used in the train and test splits:
```shell
cenfind prepare /path/to/dataset --splits 0 1 
```
The program will create two files, `test.txt` and `train.txt`, each containing a list of pairs field - channel.

After the subprogram is run, the project folder has the following structure:

```text
/<project_name>/
├── fields.txt
├── logs
├── predictions
├── projections
├── statistics
├── *test.txt
├── *train.txt
├── vignettes
└── visualisations
```

The fields.txt file maps the field names found in the raw/projections folder. They differ from the actual file names, as they are stripped of the file extension and the possible max suffix (`_max` or `_Projected`). 
The  logs/ folderwill contain the logs for the subsequent runs.
The predictions folder contains two subfolders (nuclei and centrioles) which hold the files of the predictions.
The statistics folder contains data extracted from the FOV; scores_df.tsv merely lists the centriole position, the channel it was found and the nucleus it belongs to.
The train.txt and test.txt files will only appear if the `--splits` flag is specified (see below). There are used if the a model needs to be run using the dataset, for reproducibility.
The vignettes folder is empty and will contain RGB png of each channel and the nucleus channel. Those images can be uploaded to a labelling plateform for model training.
The visualisations folder is the log of the scoring using images. You can evaluate the scoring and take the appropriate measures accordingly. 


## Computing the z-max projections with `cenfind squash`
Run `squash` with the path to the project folder and the suffix of the raw files. `projections/` is populated with the max-projections files.
```shell
cenfind squash <path/to/dataset>
```
## Scoring the FOVs
Run `score` with the arguments source, the index of the nuclei channel (usually 0 or 3), the channel to score and the path to the model, which you need to download from https://figshare.com/articles/software/Cenfind_model_weights/21724421
```shell
cenfind score /path/to/dataset /path/to/model/ 0 1 2 3
```
Once the scoring is finished, inspect `visualisations/` and `statistics/` for evaluation.

The scoring may take long (1-2 h). If you plan to log out of the shell, please read the next section.

### Running `cenfind score` in the background

When you exit the shell, running programs receive the SIGHUP, which aborts them. This is undesirable if you need to close your shell for some reasons. Fortunately, you can make your program ignore this signal by prepending the program with the `nohup` command. Moreover, if you want to run your program in the background, you can append the ampersand `&`. In practice, run `nohup cenfind score ... &` instead of `cenfind score ...`.

The output will be written to the file `nohup.out` and you can peek the progress by running `tail -F nohup.out`, the flag `-F` will refresh the screen as the file is being written. Enter Ctrl-C to exit the tail program.

If you want to kill the program score, run  `jobs` and then run `kill <jobid>`. If you see no jobs, check the log `nohup.out`; it can be done or the program may have crashed, and you can check the error there.

## Aggregating the scoring

The scoring subprogram is designed to detect both centrioles and nuclei.
It then group neighbouring centrioles into centrosome and assign them to the nearest nucleus, if it is within a radius specified by the `--vicinity` option.
Pleas note that the radius is signed and that it needs to be negative, if you tolerate the centrosome to be located outside of the nucleus boundary. If positive, the centrosome needs to be found within the nucleus boundary.

## Internal API

`cenfind` consists of two core classes: `Dataset` and `Field`.

A `Dataset` represents a collection of related fields, i.e., same pixel size, same channels, same cell type.

It is responsible for:
- returning the name
- iterating over the fields,
- constructing the file name for the projections and the z-stacks
- reading in the fields.txt
- writing the fields.txt file
- setting up the folders projections, predictions, visualisations and statistics
- setting and getting the splits

A `Field` represents a FOV and is responsible for:

- constructing file names for projections, annotation
- getting the parent Dataset
- loading the projection as np.ndarray
- loading the channel as np.ndarray
- loading annotation as np.ndarray
- loading mask as np.ndarray

Around these classes, cenfind implements detection functions in `cenfind.core.detectors`, cellular structure representations in `cenfind.core.outline` and measuring functions in `cenfind.core.measure`.

Using those two objects together with the abovementionned functions, `cenfind` can:

- detect centrioles (data, model) => points,
- extract nuclei (data, model) => contours,
- assign centrioles to nuclei (contours, points) => pairs
- outline centrioles and nuclei (data, points) => image
- create composite vignettes (data) => composite_image
- flag partial nuclei (contours, tolerance) => contours
- compare predictions with annotation (points, points) => metrics_namespace
