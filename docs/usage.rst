Usage
=====

Installation
------------

To use ReCreat, clone the repository.


Creating a land-use-based recreation model 
------------------------------------------

Create an instance of ReCreat 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, ReCreat needs to be imported and a new model instantiated.
As part of this, the working directory needs to be set.

.. note::
    The working directory is the folder in which specific scenarios are stored.
    Each scenario comprises, at the least, a land-use raster, and may furthermore 
    include scenario-specific, gridded population.  

.. code-block::
    from ReCreat import ReCreat
    working_dir = "path/to/working/directory"
    my_model = ReCreat(working_dir)

Define a scenario and import the corresponding land-use raster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To import a land-use raster for a given scenario, the scenario root folder and the name of the 
land-use raster file need to be provided, using the function:

.. autofunction:: ReCreat.set_land_use_map

