
Detect clumps
=============

Clumps are conceptualized as contiguous areas of land, determined from the gridded land-use data. Clumps provide the bounds for the 
estimation of supply, diversity, demand, flows, and cost, i.e., all indicators are bound to or constrained by individual clumps. 
Clumps are separated from each other by nodata values or so-called barrier classes, e.g., water bodies. 

.. note::
    
    Clumps will be written to disk as raster file clumps.tif in the MASKS folder.

Depending on whether recreat is used from the CLI through recreat_util, or in a script, land-use data is imported as follows.

Using recreat_util from the CLI
-------------------------------

When using recreat_util, clumps are determined through the ``clumps`` subcommand. This subcommand has the following options:

--barrier-classes        List of comma-separated barrier classes (optional, by default [0]).

Example:

.. code-block::
    
    recreat_util [...] clumps --barrier-classes 39,44

As part of calling recreat using recreat_util, clumps are detected through the ``clumps`` subcommand. 
In the example, clumps will be separated by the land-use classes 39 and 44, that are defined as barrier classes using the ``--barrier-classes`` option.   


In a script 
-----------

In a script, clump detection is conducted using the :py:meth:`.detect_clumps` method. The method has an optional ``barrier_classes`` parameter, that takes a list of barrier classes.

Example:

.. code-block:: python
    
    # define land-use classes 39 and 44 as barriers
    my_model.detect_clumps(barrier_classes=[39,44])

Clumps are detected through the :py:meth:`.detect_clumps` method. In the example, clumps will be separated by the land-use classes 39 and 44, 
that are defined as barrier classes using the ``barrier-classes`` argument.   
