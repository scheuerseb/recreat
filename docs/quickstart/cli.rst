Using recreat from the command-line
===================================

The recreat package brings its own CLI tool ``recreat-util``. Using this tool, 
a recreat model can be defined at the CLI, and package methods used readily. 

Type:

.. code-block::
    
    recreat-util --help

to see recreat_util options as well as subcommands and their options. Also refer to the list of 
subcommands and options below.

recreat_util command
--------------------

.. code-block::
    
    recreat_util [options] <root-path> [subcommand [options] <args>] [...]

Arguments:

* root-path ― The root path to use which contains relevant raster files. 

Options:

-w, --data-path                     Set path to data-path folder.
-v, --verbose                       Enable verbose output.
--datatype <int|float|double>       Specify the default datatype to be used. One of int (int32), float (float32), or double (float64).


use subcommand
--------------

.. code-block::
    
    use [options] <land-use-filename>

Arguments:

* land-use-filename ― The land-use raster file to use.

Options:

-m, --nodata         Comma-separated list of nodata values.
-f, --fill           Fill value to replace nodata values.

params subcommand
-----------------

.. code-block::

    params [options]

Options:

-p, --patch          Comma-separated list of patch classes.
-e, --edge           Comma-separated list of edge classes.
-g, --buffer-edge    Comma-separated list of buffered-edge classes.
-b, --built-up       Comma-separated list of built-up classes.
-c, --cost           Comma-separated list of cost thresholds (in pixel units, odd integer values).


reclassify subcommand
---------------------

.. code-block::

    reclassify [options] <source-classes> <destination-class>

Arguments:

  * source-classes ― Comma-separated list of source class values.
  * destination-class ― Destination class value.

Options:

-e, --export         Export recategorized raster to specified filename (by default, None).  

clumps subcommand
-----------------

.. code-block::

    clumps [options]

Options:

--barrier-classes        List of comma-separated barrier classes (optional, by default [0]).

mask-landuses subcommand
------------------------

.. code-block::

    mask-landuses

This subcommand has no arguments or options.

detect-edges subcommand
-----------------------

.. code-block::

    detect-edges [options]


Options:

-i, --ignore            Ignore edges to the specified land-use class value.

class-total-supply subcommand
-----------------------------

.. code-block::

    class-total-supply [options]

Options:

-m, --mode <generic_filter|convolve|ocv_filter2d>      Determines the moving window method to use for class supply estimation, either using SciPy's ndimage generic_filter method with a low-level callable (generic_filter), using SciPy's ndimage convolve method (convolve), or using OpenCV's filter2d method (ocv_filter2d; optional, by default, set to ocv_filter2d).


aggregate-total-supply subcommand
---------------------------------

.. code-block::

    aggregate-total-supply [options]

Options:

--landuse-weights              A comma-separated list of class values and weights, in the form class1=weight1,class2=weight2,... (optional).
-u, --exclude-non-weighted     Optional flag to indicate that the non-weighted result should not be determined.

average-total-supply subcommand
-------------------------------

.. code-block::

    average-total-supply [options]


Options:

--landuse-weights            A comma-separated list of class values and weights, in the form class1=weight1,class2=weight2,... (optional).
--cost-weights               A comma-separated list of cost thresholds and weights, in the form cost1=weight1,cost2=weight2,... (optional).
-s, --exclude-scaled         Optional flag to indicate that the scaled result should not be determined.
-u, --exclude-non-weighted   Optional flag to indicate that the non-weighted result should not be determined.

class-diversity subcommand
--------------------------

.. code-block::

    class-diversity

This subcommand has no arguments or options.

average-diversity subcommand
----------------------------

.. code-block::

    average-diversity [options]

Options:

--cost-weights                 A comma-separated list of cost thresholds and weights, in the form cost1=weight1,cost2=weight2,... (optional);
-s, --exclude-scaled           Optional flag to indicate that the scaled result should not be determined;
-u, --exclude-non-weighted     Optional flag to indicate that the non-weighted result should not be determined.

proximities subcommand
----------------------

.. code-block::

    proximities [options]

-m, --mode <dr|xr>         Method to use for determining distance (proximity) rasters, either using the distancerasters package (dr) or xarray-spatial (xr; by default, set to xr).
-b, --include-builtup      Optional flag to indicate that also proximities to built-up areas should be determined, in addition to proximities to recreational opportunities.

cost subcommand
---------------


.. code-block::

    cost [options]

Options:

-d, --max-distance        Maximum cost value used for masking of cost rasters. If set to a negative value, do not mask areas with costs higher than maximum cost. Defaults to -1.
-b, --mask-built-up       Indicates whether outputs will be restricted to built-up land-use classes, defaults to False.
-s, --exclude-scaled      Optional flag to indicate that the scaled result should not be determined.
