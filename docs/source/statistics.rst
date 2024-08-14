Statistics
==========

CenFind primary goal is the extraction of the foci and their assignement to nuclei. The resulting data structures are best processed in independent analysis custom pipelines. However, CenFind provides first-help processed results such as scores and score distribution.

Scores
------

From the assignment matrix, the scores for each nucleus can be obtained.

Linking centrioles to nuclei
----------------------------

Conversely, one can obtain the identity of the nucleus by looking up the index of the nucleus.

Score distribution
------------------

The scores for each dataset can be summarised with a histogram. For rapid checks, one can examine the statistics/statistics.tsv file:

.. csv-table:: Snippet of a centriole file
    :file: statistics.csv
    :header-rows: 1

Experimental plotting
---------------------

You can run cenfind analyse to compute some plots about the distribution of nuclei intensities, surface area / intensity.

.. code-block:: shell

    cenfind analyse path/to/dataset/

This program will save the resulting plots in png under visualisation/.