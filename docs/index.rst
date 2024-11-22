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
   Using recreat from scripts <quickstart/scripting>

.. toctree::
   :hidden:
   :caption: Fundamentals

   Prerequisites <fundamentals/prerequisites>
   Workflows <fundamentals/workflows>

.. toctree::
   :hidden:
   :caption: Usage

   Instantiation <usage/instantiation>
   Set model parameters <usage/parameters>


.. toctree::
   :hidden:
   :caption: more

   API <api>