API
===

Data
----

Manage the two level of the file system, that is, the Dataset class to set up the subdirectories and the Field class that loads the image and specify how to operate on it.

.. automodule:: src.cenfind.core.data
    :members:
    :noindex:

Structures
----------

These objects represent the different organelles to be detected.

.. automodule:: cenfind.core.structures
    :members:

Detectors
---------

Top level functions to detect specific organelles.

.. automodule:: cenfind.core.detectors
    :members:

Measure
-------
Measurement modules to analyse and combine the different objects detected.

.. automodule:: cenfind.core.measure
    :members:

Serialisers
-----------

Top level functions to write results to the file system.

.. automodule:: cenfind.core.serialise
    :members:

Statistics
----------

Top level functions to derive summary statistics from the results.

.. automodule:: cenfind.core.statistics
    :members:

Visualisation
-------------

Top level functions to draw detected objects on a variety of background images.

.. automodule:: cenfind.core.visualisation
    :members:
