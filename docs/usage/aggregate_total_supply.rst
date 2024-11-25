Aggregate total supply
============================

Aggregated total supply describes the total number of recreational opportunities (and therefore, total recreational area) 
as the sum of all relevant land-use classes within a given cost threshold. Aggregated class total supply is determined for each
cost threshold. 

.. note::
    
    Outputs (both non-weighted and/or weighted) are written into the INDICATORS subfolder. A respective raster will be exported per cost threshold.

Depending on whether recreat is used from the CLI through recreat_util, or in a script, aggregated class total supply is determined as follows.

Using recreat_util from the CLI
-------------------------------

When using recreat_util, aggregated class total supply is determined through the ``aggregate-total-supply`` subcommand. 
This subcommand has the following options:

--landuse-weights              A comma-separated list of class values and weights, in the form class1=weight1,class2=weight2,... (optional).
-u, --exclude-non-weighted     Optional flag to indicate that the non-weighted result should not be determined.

Example:

.. code-block::
    
    recreat_util [...] aggregate-total-supply --landuse-weights 24=0.3,25=0.3,26=0.3,39=0.05,44=0.05 [...]

As part of calling recreat using recreat_util, in this example, class total supply is aggregated through the 
``aggregate-total-supply`` subcommand. Here, both the non-weighted as well as the class-weighted aggregated total 
supply will be determined.  


In a script
-----------

In a script, aggregated class total supply is determined through the :py:meth:`.aggregate_class_total_supply` method. 

Example:

.. code-block:: python

    # in the model, patch classes 24, 25, and 26 were defined
    # in the model, edge classes 39 and 44 were defined
    # define weights to be used
    weights = {24 : 0.3, 25 : 0.3, 26 : 0.3, 39 : 0.05, 44 : 0.05}  
    my_model.aggregate_class_total_supply(lu_weights=weights)

In this example, class total supply is aggregated through the :py:meth:`.aggregate_class_total_supply` method. Here, 
both the non-weighted as well as the class-weighted aggregated total supply will be determined.  