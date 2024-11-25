Average cost to closest
=======================

Cost to closest describes the cost (here, distance) to the nearest opportunity of a given land-use class. 
Average cost to closest is the mean cost to all recreational opportunities that are accessible (i.e., within a maximum cost) 
from a given location (pixel).

.. note::

    The minimum cost to the nearest opportunities per land-use class is written into the COSTS subfolder. Average cost to closest is written into the INDICATORS subfolder.

Depending on whether recreat is used from the CLI through recreat_util, or in a script, (average) 
cost to closest is determined as follows.

Using recreat_util from the CLI
-------------------------------

When using recreat_util, (average) cost to closest is determined through the ``cost`` subcommand. 


In a script
-----------

In a script, (average) cost to closest is determined through the :py:meth:`.cost_to_closest` method. 

Example:

.. code-block:: python
    
    my_model.cost_to_closest()

In the example, (average) cost to closest is determined through the :py:meth:`.cost_to_closest` method.