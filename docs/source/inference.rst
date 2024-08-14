Prediction
==========


Inputs
------

The input data should be TIF images with a pixel size of about 100 nm. This constraint is due to the canonical model used in SpotNet and in StarDist. The data consists of a set of fields of view containing at least one nucleus channel and at least one centriole marker channel. The field of view should be max-projected along the depth (z-axis) and saved under a directory called ``projections/`` under the dataset directory. The file name serves as the key for the rest of the analysis.

Centriole predictions with SpotNet
----------------------------------

Cenfind uses in the background the multi-scale U-Net neural network SpotNet, which accepts a single channel max projection and predict the position of the centrioles in the field of view.
Cenfind saves a TSV file for each field of view with the position of the detected centrioles and the channel used as well as the maximum intensity at the position.

.. csv-table:: Snippet of a centriole file
    :file: snippet_centrioles.csv
    :header-rows: 1

Nuclei predictions with StarDist
--------------------------------
For nuclei detection, Cenfind uses in the background the neural network StarDist. This model takes in a single channel image of nuclei and returns a binary mask, from which the contours are extracted and saved as a JSON file for each field of view.
Each nucleus contour is saved as an entry in the JSON together with the channel index, the position (row-major), the total intensity, the surface area and a boolean variable whether the nucleus is fully in the field of view.

.. code-block:: json


  "0": {
    "channel": 0,
    "pos_r": 872,
    "pos_c": 1744,
    "intensity": 4767958,
    "surface_area": 36696,
    "is_nucleus_full": true,
    "contour": [
      [
        [
          1712,
          760
        ]
      ],
      [
        [
          1712,
          767
        ]
      ],
    ]
  },
  "1": {
    "channel": 0,
    "pos_r": 1058,
    "pos_c": 501,
    "intensity": 4270630,
    "surface_area": 33063,
    "is_nucleus_full": true,
    "contour": [
      [
        [
          496,
          936
        ]
      ],
      [
        [
          496,
          943
        ]
      ],
    ]
  },

Assignment with OR-Tools
------------------------
The detected centrioles are then assigned to the nearest nucleus provided that they lie within the set vicinity (default: 50 px).
The assigner computes an assignment table using the linear solver from the Google OR-Tools package, which extend the linear assignment to multiple jobs per agent.
Cenfind then saves this assignment matrix in the directory assignment.

Below is an example of a full assignment matrix for one field of view:

.. include:: assignment.txt
    :literal:

This NxC matrix contains row indices for nucleus ID and column indices for the centriole indices.
It describes which centrioles are assigned to which nucleus.
One can compute the number of centrioles by cell by summing over the columns and then by looking up the row number of the entry for a given centriole.
