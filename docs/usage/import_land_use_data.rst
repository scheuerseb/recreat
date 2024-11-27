Import land-use data
====================

Land-use is imported into the model from a .tif file source. As outlined here, the file needs 
to be located in a specified ``root-path`` (that may hold data specific to a given scenario, for example), 
that is in turn located within the ``data-path``. During import, grid values to be treated as 
nodata values can be specified, and a fill value to replace nodata values be specified. 

Depending on whether recreat is used from the CLI through recreat_util, or in a script, 
land-use data is imported as follows.

Using recreat_util from the CLI
-------------------------------

When using recreat_util, the root-path is specified as argument of the CLI recreat_util command. 
The land-use raster to use is then specified through the ``use`` subcommand. This subcommand has the following options:

-m, --nodata         Comma-separated list of nodata values.
-f, --fill           Fill value to replace nodata values.

.. code-block::

    recreat_util [options] root-path use [options] landuse-filename [...]


Example:

.. code-block::
    
    recreat_util [...] -m 0.0,-127.0 -f 0 current U2018_CLC2018_V2020_20u1.tif [...]

As part of calling recreat using recreat_util, this will import "U2018_CLC2018_V2020_20u1.tif" located in the 
"current" folder within the data path, treating values of 0 and -127 as nodata values, and 
replacing all nodata values with the fill value of 0. 

In a script
-----------

In a script, a landuse-file is imported using the :py:meth:`.set_land_use_map` method. 

Example:

.. code-block:: python

    my_model.set_land_use_map(
    root_path='current', 
    land_use_filename='U2018_CLC2018_V2020_20u1.tif', 
    nodata_values=[-127.0, 0.0], 
    nodata_fill_value=0)

This will import "U2018_CLC2018_V2020_20u1.tif" located in the "current" folder within the data path, treating values of 0 and -127 as nodata values, 
and replacing all nodata values with the fill value of 0. 