![alt text](figures/logos/cenfind_logo_full_dark.png)

# CenFind

**Cenfind** is a command line interface written in Python to detect and assign centrioles in immunofluorescence images of human cells.
Specifically, it orchestrates the detection of centrioles, the detection of the nuclei and the assignment of the centrioles to the nearest nucleus.

## Getting started

1. Install cenfind from PyPI:
```shell
pip install cenfind
```
2. You need to download it from https://figshare.com/articles/software/Cenfind_model_weights/21724421
3. Collect all images in a project folder inside a projections folder (<project_name>/projection/).
4. Run `score` with the path to the project, the path to the model, the index of the nuclei channel (usually 0 or 3),
   the channel to score:

```shell
cenfind score /path/to/dataset /path/to/model/ -n 0 -c 1 2 3
```

5. Check that the predictions in the folders `visualisations/` and `statistics/`

For more information, please check the documentation (https://cenfind.readthedocs.io).

## Citation

We appreciate citations as they help us obtain grant funding and let us discover its application range.

To cite Cenfind in publications, please use:

Bürgy, L., Weigert, M., Hatzopoulos, G. et al. CenFind: a deep-learning pipeline for efficient centriole detection in
microscopy datasets. BMC Bioinformatics 24, 120 (2023). https://doi.org/10.1186/s12859-023-05214-2
