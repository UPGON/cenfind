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
```shell
usage: CENFIND prepare [-h] [--projection_suffix PROJECTION_SUFFIX] [--splits SPLITS [SPLITS ...]] dataset

positional arguments:
  dataset               Path to the dataset

optional arguments:
  -h, --help            show this help message and exit
  --projection_suffix PROJECTION_SUFFIX
                        Suffix indicating projection, e.g., `_max` or `Projected`, empty if not specified (default: )
  --splits SPLITS [SPLITS ...]
                        Write the train and test splits for continuous learning using the channels specified (default: None)
```

2. Run `squash` with the path to the project folder and the suffix of the raw files. `projections/` is populated with the max-projections `*_max.tif` files.
```shell
cenfind squash path/to/dataset
```
```shell
usage: CENFIND squash [-h] path

positional arguments:
  path        Path to the dataset folder
```

3. Run `score` with the arguments source, the index of the nuclei channel (usually 0 or 3), the channel to score and the path to the model. You need to download it from https://figshare.com/articles/software/Cenfind_model_weights/21724421
```shell
cenfind score /path/to/dataset /path/to/model/ 0 1 2 3 --projection_suffix '_max'
```
```shell
usage: CENFIND score [-h] [--vicinity VICINITY] [--factor FACTOR] [--projection_suffix PROJECTION_SUFFIX] [--cpu] dataset model channel_nuclei channels [channels ...]

positional arguments:
  dataset               Path to the dataset
  model                 Absolute path to the model folder
  channel_nuclei        Channel index for nuclei segmentation, e.g., 0 or 3
  channels              Channel indices to analyse, e.g., 1 2 3

optional arguments:
  -h, --help            show this help message and exit
  --vicinity VICINITY   Distance threshold in micrometer (default: -5 um) (default: -5)
  --factor FACTOR       Factor to use: given a 2048x2048 image, 256 if 63x; 2048 if 20x: (default: 256)
  --projection_suffix PROJECTION_SUFFIX
                        Projection suffix (`_max` or `_Projected`); empty if not specified. (default: )
  --cpu                 Only use the cpu (default: False)
```

4. Check that the predictions are satisfactory by looking at the folders `visualisation/` and `statistics/`

5. If you are interested in categorising the number of centrioles, run `cenfind analyse path/to/dataset --by <well>` the --by option is interesting if you want to group your scoring by well, if the file names obey to the rule `<WELLID_FOVID>`.

```shell
usage: CENFIND analyse [-h] --by BY dataset

positional arguments:
  dataset     Path to the dataset

optional arguments:
  -h, --help  show this help message and exit
  --by BY     Grouping (field or well) (default: None)
```

## Running `cenfind score` in the background

When you exit the shell, running programs receive the SIGHUP, which aborts them. This is undesirable if you need to close your shell for some reasons. Fortunately, you can make your program ignore this signal by prepending the program with the `nohup` command. Moreover, if you want to run your program in the background, you can append the ampersand `&`. In practice, run `nohup cenfind score ... &` instead of `cenfind score ...`.

The output will be written to the file `nohup.out` and you can peek the progress by running `tail -F nohup.out`, the flag `-F` will refresh the screen as the file is being written. Enter Ctrl-C to exit the tail program.

If you want to kill the program score, run  `jobs` and then run `kill <jobid>`. If you see no jobs, check the log `nohup.out`; it can be done or the program may have crashed, and you can check the error there.

## Evaluating the quality of the model on a new dataset

The initial model M is fitted using a set of five representative datasets, hereafter referred to as the standard datasets (DS1-5). 
If your type of data deviates too much from the standard dataset, M may perform less well. 

Specifically, when setting out to score a new dataset, you may be faced with one of three situations, as reflected by the corresponding F1 score (i.e., 2TP/2TP+FN+FP, TP: true positive, FP: false positive; FN: false negative): 
(1) the initial model (M) performs well on the new dataset (0.9 ≤ F1 ≤ 1); in this case, model M is used; 
(2) model M performs significantly worse on the new dataset (0.5 ≤ F1 < 0.9); in this case, you may want to consider retraining the model (see below); 
(3) the model does not work at all (0 ≤  F1 < 0.5); such a low F1value probably means that the features of the data set are too distant from the original representative data set to warrant retraining starting from M. 

Before retraining a model (2), verify once more the quality of the data, which needs to be sufficiently good in terms of signal over noise to enable efficient learning. 
If this is not the case, it is evident that the model will not be able to learn well. 
If you, as a human being, cannot tell the difference between a real focus and a stray spot using a single channel at hand (i.e., not looking at other channels), the same will hold for the model. 

To retrain the model, you first must annotate the dataset, divide it randomly into training and test sets (90 % versus 10 % of the data, respectively). 
Next, the model is trained with the 90 % set, thus generating a new model, M*. 
Last, you will evaluate the gain of performance on the new dataset, as well as the potential loss of performance on the standard datasets. 

### Detailed training procedure:
1.	Split the dataset into training (90%) and test (10%) sets, each containing one field of view and the channel to use. This helps trace back issues during the training and renders the model fitting reproducible.
```shell
```
2.	Label all the images present in training and test sets using Labelbox. To upload the images, please create the vignettes first and then upload them once you have a project set up.
```shell
cenfind vignettes /path/to/dataset
cenfind upload /path/to/dataset --env /path/to/.env
```
3.	Save all foci coordinates (x, y), origin at top-left, present in one field of view as one text file under /path/to/dataset/annotation/centrioles/ with the naming scheme <dataset_name>_max_C<channel_index>.txt.
```shell
cenfind download dataset-name --env /path/to/.env
```
4.	Evaluate the newly annotated dataset using the model M by computing the F1 score.
evaluate dataset model

```shell
usage: CENFIND evaluate [-h] [--performances_file PERFORMANCES_FILE] [--tolerance TOLERANCE] dataset model

positional arguments:
  dataset               Path to the dataset folder
  model                 Path to the model

optional arguments:
  -h, --help            show this help message and exit
  --performances_file PERFORMANCES_FILE
                        Path of the performance file, STDOUT if not specified (default: None)
  --tolerance TOLERANCE
                        Distance in pixels below which two points are deemed matching (default: 3)
```
5.	If the performance is poor (i.e., F1 score < 0.9), fit a new model instance, M*, with the standard dataset plus the new dataset (90% in each case).
6.	Test performance of model M* on the new data set; hopefully the F1 score will now be ≥ 0.9 (if not: consider increasing size of annotated data).
7.	Test performance of model M* on the standard datasets; if performance of F1* ≥ F1, then save M* as the new M (otherwise keep M* as a separate model for the new type of data set).


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
