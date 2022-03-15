# Centrack

A command line interface to score centrioles in cells.

## Introduction

Centrack is a command line interface that allow the batch processing of ome tiff files.
Currently, it orchestrates :
- the z-max projection of the raw files, 
- the detection of centrioles 
- the detection of the nuclei
- the assignment of the centrioles to the nearest nucleus.

Eventually, centrack will be living on PyPI and can be installed using pip 
install centrack.

However, currently, this method cannot be used as centrack is not publicly 
available.

Therefore, it could be installed from the private GitHub repository 
(UPGON/centrack).

However, because the dependency spotipy cannot be downloaded via pip 
normally, one needs to download the whole repository and then git clone the 
spotipy repository under src/ next to centrack.

this situation is temporary and in the near future spotipy will become a 
simple dependency of centrack. Next, when both centrack and spotipy will be 
publicly available, centrack will be downloadable directly from PyPI.


## Setting up the Python environment prior the installation

In order to minimise interference with other Python versions, we will manage 
multiple python interpreters using pyenv. Please follow instructions at 
pyenv repository for excellent documentation about installation.

Once pyenv has been installed, set up the version to be used.

1. Install pyenv and build python 3.9.5; set it as the local and the global version using
```shell
pyenv install 3.9.5
pyenv local 3.9.5
pyenv global 3.9.5
```

To isolate centrack from other projects, only run `pip install centrack`
within a virtual environment.

3. Create a virtual environment with:
```shell
python3 -m venv centrack-venv
source centrack-venv/bin/activate
```
Your prompt should now be prepended with `(centrack-venv)`.

Check that you're at the correct location (simple and recommended location 
is `cd ~`, i.e., your home folder). 

4. Install `centrack` with:
```shell
git clone git@github.com:UPGON/centrack.git
cd centrack
```

4. Add the scripts to the PATH so that `squash` and `score` can be run from 
the command line, without the need to type the whole path.
```shell
poetry install
```

5. Check that `centrack` is correctly installed; it may take a few seconds.

```shell
squash --help
```

6. In case of updates, get the last version:
```shell
git pull

```
A common session involves running `squash`, `score` and `reduce`. Below, we 
describe each program, their input, the algorithm and the expected output.

### Squashing the stacks to projections
`squash` is a program that expect a path to a dataset folder containing a single folder 
called `raw/`. Inside raw, you have put all the folders that contains *.ome.tif 
files. These will be fetched, 
squashed and saved to `projections/`, next to the `raw` folder.

squash fetch all files recursively inside raw/ that ends with *.ome.tif. The 
files are loaded using tifffile into the memory (intensive; as each file may 
be 4.2 GB in size). Each file as up to 5 dimensions (TCZYX) but so far only 
CZYX are supported by squash. The first step is to figure out the position 
of the Z-axis. Once this has been determined, the array is max-projected 
along the z-axis and the file is saved under projections/filename_max.tif, 
where filename is extracted from the original filename. This operation is 
repeated for each file in raw.

No further preprocessing is applied to the projections, for instance the bit 
depth is unchanged (16bit) and no contrast adjustment is applied.

Nevertheless, projections files need to be converted into 8bit png files, 
prior to uploading onto Labelbox plateform. Therefore, I developed another 
program called `separate`, which takes the projections and save png for each 
of the consisting channels.

*Why saving the projections in a flat structure?*

TODO


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
squash path/to/dataset
```
4. Run `score` with the arguments source and the channel to use.
```shell
score path/to/dataset --channel 1
```
5. Check that the predictions are satisfactory by looking at the folder `outlines` and at the results/scores.csv.
6. Run `reduce` to count the number of centrioles per cell. 
```shell
reduce path/to/projections
```
