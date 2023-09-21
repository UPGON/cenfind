Prediction
==========

Centriole predictions with SpotNet
----------------------------------
Cenfind saves a TSV file for each field of view with the detected centrioles and the channel used as well as the maximum intensity at the position.



Nuclei predictions with StarDist
--------------------------------
Cenfind saves a JSON file for each field of view with the detected nuclei.
Each nucleus contour is saved as an entry in the JSON together with the channel index, the position (row, col) the summed intensity, the surface area, whether the nucleus is fully in the field of view.

Assignment with OR-Tools
------------------------

Combinatorial optimization

Cenfind saves the assignment matrix in the assignment folder.

This matrix is NxC where the row indices correspond to nucleus ID and the column indices to the centriole ID. It describes which centrioles are assigned to which nucleus. One can compute the number of centrioles by cell by summing over the columns and to retrieve the nucleus ID of every centriole assigned by looking up the row number of the entry for a given centriole.
