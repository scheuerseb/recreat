Set model parameters
====================

The following key model parameters may be specified in a recrest model: 

* Patch classes (at least one required)
* Edge classes, including specifying classes for which edges should be buffered (optional)
* Built-up classes (optional)
* Cost thresholds (in pixel units, at least one required)


.. caution::

   The actual distance in terms of the width of the kernel represented  by these cost thresholds depends 
   on the land-use grid resolution. For example, for a raster with a resolution of 100m, a cost of 21 is 
   equal to a 1000m buffer around a kernel's center pixel, and a cost of 101 would represent a 5000m buffer.  


Additionally, verbose output may be requested, that will print additional information during model 
processing, and furthermore, depending on input data (i.e., raster extent and resolution) used, 
a specific data type to be used during processing may be requested.  

Depending on whether recreat is used from the CLI through recreat_util, 
or in a script, parameters are set as follows.

Using recreat_util from the CLI
-------------------------------

Setting key model parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using recreat_util, key model parameters are set using the ``params`` subcommand. 
This subcommand has the following options:

-p, --patch            Comma-separated list of patch classes.
-e, --edge             Comma-separated list of edge classes.
-g, --buffered-edge    Comma-separated list of buffered-edge classes.
-b, --built-up         Comma-separated list of built-up classes.
-c, --cost             Comma-separated list of cost thresholds (in pixel units, odd integer values).

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

In a script, model parameters are set using the :py:meth:`.set_params` method.
Supported ``param_name`` and ``param_value`` area:

=================     ===============
Parameter name        Parameter value
=================     ===============
classes.patch         Comma-separated list of patch classes.
classes.edge          Comma-separated list of edge classes.
classes.builtup       Comma-separated list of built-up classes.
costs                 Comma-separated list of cost thresholds (in pixel units, odd integer values).
use-data-type         Datatype to be used, one of ``np.int32``, ``np.float32``, or ``np.float64``.
verbose-reporting     If set to True, enable verbose reporting.
=================     ===============

.. note::

   Note that unlike at the CLI, edge classes for which edges should be buffered are not specified at the level of model parameters, 
   but as parameter to the detect_edges method. Therefore, in a script, the list of the classes.edge parameter should 
   include the complete set of desired edge classes, including buffered-edge classes. 

Example:

.. code-block:: python

   # numpy needs to be imported to specify the datatype
   import numpy as np

   my_model.set_params('classes.patch', [24, 25, 26]) 
   my_model.set_params('classes.edge', [39, 44])
   my_model.set_params('classes.builtup', [1, 2])
   my_model.set_params('costs', [21, 101])
   my_model.set_params('verbose-reporting', True)
   my_model.set_params('use-data-type', np.int32)

Model parameters are set using the set_params method. Here, grid values of 24, 25, or 26 will be treated as patch classes. 
Edge classes include grid values 39 and 44 (however, unlike at the CLI, setting class 44 as buffered-edge class will be conducted 
in the corresponding :py:meth:`.detect_edges` method). Built-up area is defined by classes 1 and 2, and two cost thresholds of 21 pixels and 
101 pixels will be assessed. 