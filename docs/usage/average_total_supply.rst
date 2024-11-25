Average total supply
====================

The averaged total supply corresponds to the sum of class total supplies across cost thresholds, 
divided by the number of cost thresholds considered. Depending on the options/parameters provided, the following outputs are provided:

* Non-weighted average total supply: Simple average of class total supply across cost thresholds (if not specifically excluded from the result set);
* Class-weighted average total supply: Land-use-weighted average of class total supply across cost thresholds (if land-use weights, i.e., a weight provided for each land-use class, has been specified); 
* Cost-weighted average total supply: Cost-weighted average of class total supply across cost thresholds (if cost weights, i.e., a weight provided for each cost threshold, have been specified);
* Bi-weighted average total supply: Land-use- and cost-weighted average of class total supply across cost thresholds (if both land-use weights and cost weights have been specified);

For each of these outputs, a scaled (i.e., min-max normalized raster in the range [0;1]) raster will be exported (if not specifically excluded 
from the result set). 

.. note::
    
    All outputs will be written to the INDICATORS subfolder.

Depending on whether recreat is used from the CLI through recreat_util, or in a script, average total supply is determined as follows.


Using recreat_util from the CLI
-------------------------------

When using recreat_util, average total supply is determined using the ``average-total-supply`` subcommand. 
This subcommand has the following options:

--landuse-weights            A comma-separated list of class values and weights, in the form class1=weight1,class2=weight2,... (optional).
--cost-weights               A comma-separated list of cost thresholds and weights, in the form cost1=weight1,cost2=weight2,... (optional).
-s, --exclude-scaled         Optional flag to indicate that the scaled result should not be determined.
-u, --exclude-non-weighted   Optional flag to indicate that the non-weighted result should not be determined.

Example:

.. code-block::
    
    recreat_util [...] average-total-supply --cost-weights 21=0.8,101=0.2 -u [...]

As part of calling recreat using recreat_util, in this example, average total supply is determined through the ``average-total-supply`` 
subcommand. Through the --cost-weights option, weights are defined for each cost threshold; accordingly, cost-weighted average total supply will be exported. The non-weighted average total supply will not be determined, as the -u flag has been set. 
subcommand. 

In a script
-----------

In a script, average total supply is determined through the :py:meth:`.average_total_supply_across_cost` method. 

Example:

.. code-block:: python
    
    # in the model, cost thresholds 21 and 101 were defined
    # define cost weights to be used
    cost_weights = {21 : 0.8, 101 : 0.2}
    my_model. average_total_supply_across_cost(
    cost_weights=cost_weights,
    write_non_weighted_result=False)

In this example, average total supply is determined through the average_total_supply_across_cost method. A weight is defined 
for each cost threshold; accordingly, cost-weighted average class total supply will be exported. 
The non-weighted average total supply will not be determined, as the write_non_weighted_result 
parameter has been set to False. 