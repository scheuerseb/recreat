Set model parameters
====================

The following key model parameters may be specified in a recrest model: 

* Patch classes (at least one required)
* Edge classes, including specifying classes for which edges should be buffered (optional)
* Built-up classes (optional)
* Cost thresholds (in pixel units, at least one required)

Additionally, verbose output may be requested, that will print additional information during model 
processing, and furthermore, depending on input data (i.e., raster extent and resolution) used, 
a specific data type to be used during processing may be requested.  

Depending on whether recreat is used from the CLI through recreat_util, 
or in a script, land-use data is imported as follows.

Using recreat_util from the CLI
-------------------------------

Setting key model parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using recreat_util, key model parameters are set using the ``params`` subcommand. 
This subcommand has the following options:

-p, --patch          Comma-separated list of patch classes.
-e, --edge           Comma-separated list of edge classes.
-g, --buffer-edge    Comma-separated list of buffered-edge classes.
-b, --built-up       Comma-separated list of built-up classes.
-c, --cost           Comma-separated list of cost thresholds (in pixel units, odd integer values).

.. hint::

   The total set of edge classes comprises all classes specified as edge and buffered-edge classes.

Example:

.. code-block::
   
   recreat_util [...] params -p 24,25,26 -e 39 -g 44 -b 1,2 -c 21,101 [...]

As part of calling recreat using recreat_util, model parameters are set using the params subcommand. 
Here, grid values of 24, 25, or 26 will be treated as patch classes. Edge classes include grid values 
39 and 44, however, for the class values of 44, edges will be buffered. 
Built-up area is defined by classes 1 and 2, and two cost thresholds of 21 pixels and 101 pixels 
will be assessed. 

.. caution::

   The actual distance in terms of the width of the kernel represented  by these cost thresholds depends 
   on the land-use grid resolution. For example, for a raster with a resolution of 100m, a cost of 21 is 
   equal to a 1000m buffer around a kernel's center pixel, and a cost of 101 would represent a 5000m buffer.  

Setting additional model parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Additional parameters include verbosity and specifying an explicit datatype to use. For convenience, 
at the CLI level, additional model parameters are implemented as options at the recreat_util command-level, 
as follows:

-v, --verbose                       Enable verbose output.
--datatype <int|float|double>       Specify the default datatype to be used. One of int (int32), float (float32), or double (float64).



Example:

.. code-block::
   
   recreat_util -v --datatype float [...] 

As part of calling recreat using recreat_util, verbose output is enabled, and float32 is set as 
default datatype. 


In a script
-----------

