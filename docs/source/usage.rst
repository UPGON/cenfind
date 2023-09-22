Usage
=====

.. _installation:

Installation
------------

To use Cenfind CLI, just install it from pip

.. code-block:: console

   (.venv) % pip install cenfind

Basic usage
-----------

``cenfind`` assumes a fixed folder structure.
Specifically, it expects the max-projection to be under the ``projections`` folder.
Each file in projections is a z-max projected field of view (referred to as field, in the following) containing 4
channels (0, 1, 2, 3). The channel 0 usually contains the nuclei and the channels 1-3 contains centriolar markers.

1. Run ``score`` and indicate the path to the dataset directory, the path to the model (https://figshare.com/articles/software/Cenfind_model_weights/21724421) followed by the index of the nuclei channel (-n) and the channels to score (-c).

In the following example

.. code-block:: shell

    cenfind score /path/to/dataset /path/to/model/ -n 0 -c 1 2 3

2. Check that the predictions are satisfactory by looking at the folders ``visualisations/`` and ``statistics/``

.. versionadded:: 0.13.0
    The outputs can be linked together depending on the applications. Detailed explanation are there :doc:`inference`.


Summary statistics
------------------

Besides the raw inference data (centriole position and intensity information, nuclei contours and geometry information), the statistics folder contains precomputed information about the distribution of centriole number (statistics.tsv), TSV files for pairs of assigned centrioles their nucleus if possible. If the cilia are analysed, a TSV file containing the fraction of ciliated cells is saved as well. You can read more about statistics here :doc:`statistics`.

Running ``cenfind score`` in the background
-------------------------------------------

When you exit the shell, running programs receive the SIGHUP, which aborts them. This is undesirable if you need to
close your shell for some reason. Fortunately, you can make your program ignore this signal by prepending the program
with the ``nohup`` command. Moreover, if you want to run your program in the background, you can append the ampersand ``&``.
In practice, run ``nohup cenfind score ... &`` instead of ``cenfind score ...``.

The output will be written to the file ``nohup.out`` and you can peek the progress by running ``tail -F nohup.out``, the
flag ``-F`` will refresh the screen as the file is being written. Enter Ctrl-C to exit the tail program.

If you want to kill the program score, run  ``jobs`` and then run ``kill <jobid>``. If you see no jobs, check the
log ``nohup.out``; it can be done or the program may have crashed, and you can check the error there.
