recreat outputs
===============

When instantiating a model, recreat creates a set of folders within the model's ``root-path``.
Subsequently, recreat outputs will be written into these folders. 

==========       =================
Folder           Description
==========       =================
MASKS            Land-use class masks, edges and clump raster
SUPPLY           Class total supply
DIVERSITY        Class diversity
INDICATORS       Aggregated or averaged indicators (total supply, diversity, cost)
PROX             Distance (proximity) raster
DEMAND
FLOWS
TMP              Temporary files will be written into this folder.
==========       =================

.. note::

    Temporary files are automatically deleted after model completion.