Import land-use data
=====================

Land-use is imported into the model from a .tif file source. The file needs to be located in a so-called ``root-path``, 
i.e., a folder within the ``data-path``. You may consider any given ``root-path`` as a scenario-specific folder, that holds land-use and other
data specific to a given scenario. 

A ``root-path`` and corresponding land-use file to be used in an assessment is specified through the :py:meth:`.set_land_use_map` method. 
Note that this only sets the respective environment parameters in the model, but does not import the file into the model. The latter is
conducted through the :py:meth:`.align_land_use_map` method. 

This :py:meth:`.align_land_use_map` method allows specifying values to be handled as nodata values, 
setting the corresponding band to be read from the tif file source, and also allows to reclassify values in the land-use layer. Reclassification is achieved by
providing a dictionary of mappings (type ``Dict[int, List[int]]``) to the ``reclassification_mappings`` argument, where the dictionary keys correspond to new (destination) class values, and the dictionary values 
to Lists of one or more (source) class values to be recategorized into the new class value.   


Examples:

The following example sets root path and land-use map, and imports the land-use map into the model: 

.. code-block:: python

    my_model.set_land_use_map(
        root_path='current', 
        land_use_filename='U2018_CLC2018_V2020_20u1.tif'
    )
    my_model.align_land_use_map()

The following example additionally conducts a reclassification of classes 810 to 850 into a new class 800:

.. code-block:: python

    my_model.set_land_use_map(
        root_path='current', 
        land_use_filename='U2018_CLC2018_V2020_20u1.tif'
    )

    mapping = {800: [810, 820, 830, 840, 850]}
    my_model.align_land_use_map(reclassification_mappings=mapping)


.. note::

   A set of folders will be created in the model's ``root-path``, into which outputs will subsequently be written.

