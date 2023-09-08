# CenFind

A command line interface to score cells for centrioles.

## Introduction

`cenfind` is a command line interface to detect and assign centrioles in immunofluorescence images of human cells.
Specifically, it orchestrates:

- the detection of centrioles;
- the detection of the nuclei;
- the assignment of the centrioles to the nearest nucleus.

You can read more on it here: Bürgy, L., Weigert, M., Hatzopoulos, G. et al. CenFind: a deep-learning pipeline for efficient centriole detection in microscopy datasets. BMC Bioinformatics 24, 120 (2023). https://doi.org/10.1186/s12859-023-05214-2

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

5. Check that `cenfind` is correctly installed by running:

```shell
cenfind --help
```

## Basic usage

`cenfind` assumes a fixed folder structure.
Specifically, it expects the max-projection to be under the `projections` folder.
Each file in projections is a z-max projected field of view (referred to as field, in the following) containing 4
channels (0, 1, 2, 3). The channel 0 usually contains the nuclei and the channels 1-3 contains centriolar markers.

```text
<project_name>/
└── projections/
```

1. Run `score` with the arguments source, the index of the nuclei channel (usually 0 or 3), the channel to score and the
   path to the model. You need to download it from https://figshare.com/articles/software/Cenfind_model_weights/21724421

```shell
cenfind score /path/to/dataset /path/to/model/ -n 0 -c 1 2 3
```

```shell
usage: CENFIND score [-h] --channel_nuclei CHANNEL_NUCLEI [--channel_centrioles CHANNEL_CENTRIOLES [CHANNEL_CENTRIOLES ...]] [--channel_cilia CHANNEL_CILIA] [--vicinity VICINITY] [--cpu] dataset model

positional arguments:
  dataset               Path to the dataset
  model                 Absolute path to the model folder

options:
  -h, --help            show this help message and exit
  --channel_nuclei CHANNEL_NUCLEI, -n CHANNEL_NUCLEI
                        Channel index for nuclei segmentation, e.g., 0 or 3 (default: None)
  --channel_centrioles CHANNEL_CENTRIOLES [CHANNEL_CENTRIOLES ...], -c CHANNEL_CENTRIOLES [CHANNEL_CENTRIOLES ...]
                        Channel indices to analyse, e.g., 1 2 3 (default: [])
  --channel_cilia CHANNEL_CILIA, -l CHANNEL_CILIA
                        Channel indices to analyse cilium (default: None)
  --vicinity VICINITY, -v VICINITY
                        Distance threshold in pixel (default: 50 px) (default: 50)
  --cpu                 Only use the cpu (default: False)
```

2. Check that the predictions are satisfactory by looking at the folders `visualisations/` and `statistics/`

## The outputs in version 0.13.x

In version 0.13, we operated a shift in what cenfind-score outputs. Now, there are modular outputs that can be linked together depending on the applications. In the following section, each output is explained.

### Assignment

Cenfind saves the assignment matrix in the assignment folder.

This matrix is NxC where the row indices correspond to nucleus ID and the column indices to the centriole ID. It describes which centrioles are assigned to which nucleus. One can compute the number of centrioles by cell by summing over the columns and to retrieve the nucleus ID of every centriole assigned by looking up the row number of the entry for a given centriole.

### Centriole predictions
Cenfind saves a TSV file for each field of view with the detected centrioles and the channel used as well as the maximum intensity at the position.

### Nuclei predictions

Cenfind saves a JSON file for each field of view with the detected nuclei. Each nucleus contour is saved as an entry in the JSON together with the channel index, the position (row, col) the summed intensity, the surface area, whether the nucleus is fully in the field of view.

### Cilia
If specified by the user at the command line prompt, the cilia can be analysed in the given channel. In such cases, the folder called cilia will contain TSV files similar in structure to the one from centrioles.

### Summary statistics
The statistics folder contains precomputed information about the distribution of centriole number (statistics.tsv), TSV files for pairs of assigned centrioles their nucleus if possible. If the cilia are analysed, a TSV file containing the fraction of ciliated cells is saved as well.

## Running `cenfind score` in the background

When you exit the shell, running programs receive the SIGHUP, which aborts them. This is undesirable if you need to
close your shell for some reason. Fortunately, you can make your program ignore this signal by prepending the program
with the `nohup` command. Moreover, if you want to run your program in the background, you can append the ampersand `&`.
In practice, run `nohup cenfind score ... &` instead of `cenfind score ...`.

The output will be written to the file `nohup.out` and you can peek the progress by running `tail -F nohup.out`, the
flag `-F` will refresh the screen as the file is being written. Enter Ctrl-C to exit the tail program.

If you want to kill the program score, run  `jobs` and then run `kill <jobid>`. If you see no jobs, check the
log `nohup.out`; it can be done or the program may have crashed, and you can check the error there.

## Evaluating the quality of the model on a new dataset

The initial model M is fitted using a set of five representative datasets, hereafter referred to as the standard
datasets (DS1-5).
If your type of data deviates too much from the standard dataset, M may perform less well.

Specifically, when setting out to score a new dataset, you may be faced with one of three situations, as reflected by
the corresponding F1 score (i.e., 2TP/2TP+FN+FP, TP: true positive, FP: false positive; FN: false negative):
(1) the initial model (M) performs well on the new dataset (0.9 ≤ F1 ≤ 1); in this case, model M is used;
(2) model M performs significantly worse on the new dataset (0.5 ≤ F1 < 0.9); in this case, you may want to consider
retraining the model (see below);
(3) the model does not work at all (0 ≤ F1 < 0.5); such a low F1-value probably means that the features of the data set
are too distant from the original representative data set to warrant retraining starting from M.

Before retraining a model (2), verify once more the quality of the data, which needs to be sufficiently good in terms of
signal over noise to enable efficient learning.
If this is not the case, it is evident that the model will not be able to learn well.
If you, as a human being, cannot tell the difference between a real focus and a stray spot using a single channel at
hand (i.e., not looking at other channels), the same will hold for the model.

To retrain the model, you first must annotate the dataset, divide it randomly into training and test sets (90 % versus 10 % of the data, respectively).
Next, the model is trained with the 90 % set, thus generating a new model, M*.
Last, you will evaluate the gain of performance on the new dataset, as well as the potential loss of performance on the standard datasets.

### Detailed training procedure:

1. Split the dataset into training (90%) and test (10%) sets, each containing one field of view and the channel to use.
   This helps trace back issues during the training and renders the model fitting reproducible.

2. Label all the images present in training and test sets using Labelbox. To upload the images, please create the vignettes first and then upload them once you have a project set up.
3. Save all foci coordinates (x, y), origin at top-left, present in one field of view as one text file under
   /path/to/dataset/annotation/centrioles/ with the naming scheme <dataset_name>_max_C<channel_index>.txt.
4. Evaluate the newly annotated dataset using the model M by computing the F1 score.
5. If the performance is poor (i.e., F1 score < 0.9), fit a new model instance, M*, with the standard dataset plus the
   new dataset (90% in each case).
6. Test performance of model M* on the new data set; hopefully the F1 score will now be ≥ 0.9 (if not: consider
   increasing size of annotated data).
7. Test performance of model M* on the standard datasets; if performance of F1* ≥ F1, then save M* as the new M (
   otherwise keep M* as a separate model for the new type of data set).
