Spatial disaggregation
======================

The spatial disaggregation of gridded population to built-up land-use classes, i.e., 
pixels treated as residential areas, is a key step in the estimation of recreational 
demand and (class-based) flow(s).

.. note::
    
    Disaggregated population, as well as intermediate outputs, will be written to disk in the DEMAND folder.


Using recreat_util from the CLI
-------------------------------

When using recreat_util, population is disaggregated through the ``disaggregate`` subcommand. This subcommand has the following arguments and 
options:

Arguments:

<population-grid>       The gridded population raster.

Options:

-m, --method            Disaggregation method. One of 'saw' (simple area weighted) or 'idm' (intelligent dasymetric mapping), by default 'saw'.
-c, --pixel-count       Number of built-up pixels per population grid cell.
-n, --sample-size       Minimum number of pixels required to determine class relative density (for idm only).
-t, --threshold         Minimum sampling threshold (share of class in population grid cell as source unit; for idm only).
-s, --exclude-scaled    Optional flag to indicate that the scaled result should not be determined;


Example:

.. code-block::
    
    recreat_util [...] disaggregate ...

As part of calling recreat using recreat_util, population is disaggregated through the ``disaggregate`` subcommand. 
In the example, ...

In a script 
-----------

In a script, population disaggregated is conducted using the :py:meth:`.disaggregation` method. ...
