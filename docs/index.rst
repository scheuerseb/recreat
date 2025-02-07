Welcome to recreat's documentation!
===================================

recreat (/re’creat'/) is a Python library for assessing landscape recreational potential. 
Applying the concept of ecosystem services, recreat assesses dimensions of landscape recreational 
potential through the analytical lenses of supply, demand, flow, and cost. recreat is 
following a data-driven, rapid assessment approach, allowing for flexibility and transferability.

What does recreat do?
---------------------

recreat assesses landscape recreational potential in the form of land use supply, 
heterogeneity respectively diversity, demand and flow, as well as (minimum) cost. 
Broadly, recreat:

* consumes gridded land use data to determine the presence and absence of land use classes;
* aggregates the supply (presence) of land use classes assumed to have recreational potential, based on the user's model specification, within and across user-defined cost (distance) thresholds;
* determines, based on previously estimated supply, the heterogeneity respectively diversity of land use classes with recreational potential within and across chosen cost thresholds; 
* estimates, based on gridded population data, recreational demand in terms of the potential number of beneficiaries of recreational land use supply, and furthermore estimates potential flows of beneficiaries to service-providing areas;
* computes distance (proximity) rasters to recreational opportunities, and determines (minimum) costs to nearest recreational opportunities.   


.. note::

   Please check the detailed documentation hosted `here <https://sebsc.gitbook.io/recreat/>`_ for the usage of recreat.

.. toctree::
   :hidden:
   
   Home <self>

.. toctree::
   :hidden:
   :caption: Quickstart
   
   Installation <quickstart/installation>
   Using recreat from the command-line <quickstart/cli>
   Using recreat from scripts <quickstart/scripting>
   recreat outputs <quickstart/outputs>

.. toctree::
   :hidden:
   :caption: Fundamentals

   Prerequisites <fundamentals/prerequisites>
   Workflows <fundamentals/workflows>

.. toctree::
   :hidden:
   :caption: Script Usage

   Instantiation <usage/instantiation>
   Import land-use data <usage/import_land_use_data>
   Set model parameters <usage/parameters>
   Reclassification <usage/reclassification>
   Detect clumps <usage/clumps>
   Mask land-uses <usage/masking>
   Edge detection <usage/detect_edge.rst>
   Class total supply <usage/class_total_supply>
   Aggregate total supply <usage/aggregate_total_supply>
   Average total supply <usage/average_total_supply>
   Class diversity <usage/class_diversity>
   Average diversity <usage/average_diversity>
   Compute distance rasters <usage/proximity_rasters>
   Average cost <usage/average_cost>
   Disaggregation <usage/disaggregation>

.. toctree::
   :caption: CLI Usage

   Instantiation <usage_cli/instantiation>



.. toctree::
   :hidden:
   :caption: API

   assessment <api/Recreat>
   disaggregation <api/disaggregation>
   clustering <api/clustering>