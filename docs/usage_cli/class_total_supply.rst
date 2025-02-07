Class total supply
===================

Class total supply is the total number of opportunities (respectively, the total area) of a given land-use class, 
within a given cost threshold. Therefore, class total supply is determined for each land-use class and for each cost threshold. 
Class total supply estimation requires clump detection, land-use masking and edge detection to be completed, as their outputs 
are consumed by class total supply estimation as inputs. 

.. note::
    
    Outputs will be written to the SUPPLY subfolder.

Depending on whether recreat is used from the CLI through recreat_util, or in a script, class total supply is estimated as follows.

Using recreat_util from the CLI
-------------------------------

When using recreat_util, class total supply is determined through the ``class-total-supply`` subcommand. 
This subcommand has the following options:

-m, --mode <generic_filter|convolve|ocv_filter2d>      Determines the moving window method to use for class supply estimation, either using SciPy's ndimage generic_filter method with a low-level callable (generic_filter), using SciPy's ndimage convolve method (convolve), or using OpenCV's filter2d method (ocv_filter2d; optional, by default, set to ocv_filter2d).


Example:

.. code-block::
    
    recreat_util [...] class-total-supply -m generic_filter [...]


As part of calling recreat using recreat_util, class total supply is determined through the ``class-total-supply`` subcommand. 
The method to use for moving window estimation is the generic_filter method from SciPy's ndimage. 

In a script
-----------

In a script, class total supply is determined through the :py:meth:`.class_total_supply` method.

Example:

.. code-block:: python
    
    my_model.class_total_supply(mode="generic_filter")

In the example, class total supply will be estimated using SciPy's ndimage generic_filter method.