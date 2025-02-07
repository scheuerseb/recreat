Compute distance raster
=======================

Distance rasters (or proximity rasters) are required to assess (average) cost to recreational opportunities.

.. warning::

    The calculation of distance (proximity) grids is computationally intensive and can take a considerable amount of time.

.. note::

    Outputs will be written into the PROX subfolder.

Depending on whether recreat is used from the CLI through recreat_util, or in a script, distance rasters are computed as follows.

Using recreat_util from the CLI
-------------------------------

When using recreat_util, distance rasters are determined through the ``proximities`` subcommand. 
This subcommand has the following options:

-m, --mode <dr|xr>         Method to use for determining distance (proximity) rasters, either using the distancerasters package (dr) or xarray-spatial (xr; by default, set to xr).
-b, --include-builtup      Optional flag to indicate that also proximities to built-up areas should be determined, in addition to proximities to recreational opportunities.


Example:

.. code-block::
    
    recreat_util [...] proximities -m dr [...]

As part of calling recreat using recreat_util, in this example, distance (proximity) rasters are determined through the 
``proximities`` subcommand. Using the ``-m`` option, the method to determine proximities is set to distancerasters. 
Proximities to built-up areas will not be determined.


In a script
-----------

In a script, distance (proximity) rasters are determined through the :py:meth:`.compute_distance_rasters` method. 

Example:

.. code-block:: python
    
    my_model.compute_distance_rasters(mode = 'dr')

As part of calling recreat using recreat_util, in this example, distance (proximity) rasters are determined through 
the :py:meth:`.compute_distance_rasters` method. By setting the ``mode`` parameter to 'dr', the method to determine 
proximities is set to distancerasters. Proximities to built-up areas will not be determined.