Spatial disaggregation
======================

The spatial disaggregation of gridded population to built-up land-use classes, i.e., 
pixels treated as residential areas, is a key step in the estimation of recreational 
demand and (class-based) flow(s).

.. note::
    
    Disaggregated population, as well as intermediate outputs, will be written to disk in the DEMAND folder.


Disaggregation methods

Spatial disaggregation of population refers to the spatial allocation of the population of a given source zone, e.g., a census tract, neighbourhood, 
or district, to relevant target zones, i.e., residential land-uses within the source zone in question. Here, referring to raster data, a source zone typically 
corresponds to a pixel of a population raster, whereas target zones correspond to pixels of relevant residential land-use classes. 
There are various methods to conduct such disaggregation. The recreat package currently implements two distinct spatial disaggregation methods: 
(i) Simple Area Weighting; and (ii) Intelligent Dasymetric Mapping. 

Simple Area Weighting

Using this method, population of a source zone is proportionally distributed to target zones within the source zone solely as a function of each 
source zone's area. In this case, area refers to the respective numbers of pixels of each relevant land-use class within a given source zone. 
Therefore, despite methodologically considering distinct residential land-use classes in the disaggregation, all residential land-use classes are 
considered equal in terms of population density. 

Intelligent Dasymetric Mapping

Using this method, population of a source zone is disaggregated to relevant target zones as a function of area as well as relative density of 
residential land-use classes. This relative density is estimated from provided data, considering a user-defined sampling 
threshold and a minimum sample size. Population is then estimated for each target zone as:

.. math::
    \hat{y}_{t} =
    \begin{cases}
        \frac{y_{s} D_{c,t}}{1} & \text{if $D_{c,t} > 0$},\\
    0 & \text{otherwise}
    \end{cases}

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
