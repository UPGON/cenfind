# Centrack

A command line interface to score centrioles in cells.

## Installation

```shell
pip install centrack
```

## Usage

1. Navigate to the dataset folder.
2. Group all the raw OME.TIFF files into a folder called `raw`.
This helps keeping the structure of the processed images clean.
3. centrack runs on z-max projections so before running the main program `centrack score`, you may want to
run `centrack project` to save computing time in the future.
4. After running the `centrack project`, a folder called `projections` is created and contains the 4-d tiff files.
5. Now, you can swiftly run `centrack score` on the projections files.
6. You now can specify a few options to guide the scoring.

### Options overview

* Selecting the centriole detection method
* Choosing for conservative detection or liberal
* Selecting the cell segmentation algorithm

### Architecture requirements
- Separate the z-max projection code (caching)
- Choice: lightweight stack object or use existing classes (AICSImageIO)