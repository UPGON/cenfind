How to contribute?
==================

Currently, Cenfind is not actively further developed. However, we would like to point to some directions where contributions would be most welcome and useful. Indeed, the training code is as of now to much coupled with the prediction mode of Cenfind. So one improvement would be to move the data loading functions and the management of the train/validation/test splits in a separate class (Loader). In addition, the release of Spotiflow allows one to fully transition to the PyTorch ecosystem and thus adopt the dataset/dataloader class of PyTorch. Therefore, it would not be necessary to spend to much effort on re-implementing the data loading functionality.
