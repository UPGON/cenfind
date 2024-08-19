How to contribute?
==================

Although Cenfind is currently not actively developed, it does not mean it is perfect, far from it. We would like therefore to discuss which contributions would be most impactful.

The team behind Spotipy, the centriole detector has now published the preprint of the enhanced version "Spotiflow" (https://www.biorxiv.org/content/10.1101/2024.02.01.578426v2). We encourage anyone to follow the instructions from Spotiflow (https://github.com/weigertlab/spotiflow/) to train a new model and to integrate the new model into Cenfind. Incidentally, the release of Spotiflow allows one to fully transition to the PyTorch ecosystem and thus adopt the dataset/dataloader class of PyTorch.

At publication time, we included the code to train the model and evaluate its performance. We also had written code to manage the upload of pred-annotated dataset to the platform Labelbox, as well as the code to download the corrected annotated dataset. However, the main aim of Cenfind is its application on datasets in prediction mode using an existing model. Thus, the training part has been relegated to submodules and the code is not written in a client-oriented manner. Also, the training code is somewhat coupled with the prediction mode of Cenfind. One improvement would be to move the data loading functions and the management of the train/validation/test splits in a separate class (Loader). Therefore, it would not be necessary to re-implement the data loading functionality.

On a more algorithmic point of view, Cenfind detects the contours of the nuclei and assign the centrioles to the nearest nucleus. This heuristics leads to a probability of wrong assignments. It would be interesting to detect the cell boundaries, as opposed to the detection of the nuclei.

The images accepted by the pipeline should be acquired with a given pixel size. It would be interesting to relax this constraint and accept a wider range of pixel size, especially larger pixel sizes, i.e., lower magnification, when working with screening imaging modalities.