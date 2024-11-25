Average diversity
=================

Average diversity is the average of class diversity across cost thresholds. Depending on the options/parameters provided, 
the following outputs are provided:

* Non-weighted average diversity: Simple average of class diversity across cost thresholds (if not specifically excluded from the result set);
* Cost-weighted average diversity: Cost-weighted average of class diversity across cost thresholds (if cost weights, i.e., a weight provided for each cost threshold, have been specified);
* For each of these outputs, a scaled raster, i.e., a min-max normalized raster in the range [0;1],  will be exported (if not specifically excluded from the result set).

.. note::

    All outputs will be written to the INDICATORS subfolder.

Depending on whether recreat is used from the CLI through recreat_util, or in a script, average diversity is determined as follows.

Using recreat_util from the CLI
-------------------------------

When using recreat_util, average diversity is determined through the ``average-diversity`` subcommand. 
This subcommand has the following options:

--cost-weights                 A comma-separated list of cost thresholds and weights, in the form cost1=weight1,cost2=weight2,... (optional);
-s, --exclude-scaled           Optional flag to indicate that the scaled result should not be determined;
-u, --exclude-non-weighted     Optional flag to indicate that the non-weighted result should not be determined.

Example:

.. code-block::
    
    recreat_util [...] average-diversity -s [...]

As part of calling recreat using recreat_util, in this example, average diversity is determined through the ``average-diversity`` 
subcommand. Here, as no cost weights are specified, only the non-weighted average diversity will be determined. Through setting of the -s flag, scaled outputs are not determined. 
subcommand. 

In a script
-----------

In a script, average diversity is determined through the :py:meth:`.average_diversity_across_cost` method. 

Example:

.. code-block:: python
    
    my_model.average_diversity_across_cost(write_scaled_result=False)

In this example, average diversity is determined through the average_diversity_across_cost method. 
Here, as no cost weights are specified, only the non-weighted average diversity will be determined. 
Results will not be scaled, as scaled outputs are excluded by setting the write_scaled_result parameter 
to False. 