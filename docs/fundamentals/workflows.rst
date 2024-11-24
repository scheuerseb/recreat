Workflows
=========

Model specification and computation of basic data are commonly conducted for 
each model. Depending on the indicators of interest, subsequent steps may differ. 
In the following, typical workflows and respectively, modelling steps are presented. 

Model preparation
-----------------

Model preparation includes the preparation of required data, instantiating a new recreat model, 
conceptualizing model content and setting model parameters accordingly, and importing required data, 
particularly the gridded land-use, into the model. Land-use classes may subsequently be reclassified, 
if needed. 

1. Prepare data

First, prepare the data structure required by recreat. Therefore, copy gridded land-use data 
(and if available, population data) into corresponding root-path folders within the data-path. 

2. Instantiate a new recreat model

Second, instantiate a new recreat model.

3. Set model parameters

For your model, it needs to be conceptualized which classes are relevant as patch and edge 
classes, and similarly, other classes (such as built-up) migh be included as needed, as well as 
cost thresholds of interest need to be defined. Accordingly, model parameters must be set.

4. Import land-use data

On the one hand, the gridded land-use dataset serves as reference layer, informing on the total 
extent and spatial resolution of the model.  On the other hand, this layer is crucial as it 
obviously provides the land-use classification to be assessed by recreat. Hence, importing the 
land-use data into the model is a common step for all recreat models.

5. Reclassify land-use classes

If needed, you may also reclassify land-use classes, to adapt the classes provided by the 
chosen land-use file to specific needs.  

Basic data processing
---------------------

Basic data processing are steps conducted to pre-process the input data for the subsequent 
estimation of indicators. As result of this pre-processing, various layers will be computed 
that are consumed as input data in subsequent steps, including, e.g., land-use masks and 
land-use edges.

1. Detect clumps

Clumps are contiguous areas of land (e.g., mainland Europe, British Isles, smaller islands, etc.), 
determined from the gridded land-use dataset.  Computed indicators are commonly bound to each 
identified clump. Therefore, as part of basic data processing, clumps need to be identified first, 
for subsequent steps of the analysis.

2. Mask land-uses

Land-use masking creates land-use masks (or so-called class masks), provide the basis for 
the estimation of supply.  

3. Detect edges

If the model contains land-use classes which are deemed relevant in the form of edges, their 
patch perimeters will be determined in detect edges. 


Estimate indicators of supply
-----------------------------

Based on land-use masks and clumps, class total supply is determined, which in turn allows assessing 
aggregated or averaged total supply.

1. Determine class total supply

Class total supply, i.e., the total number (or conversely, total area) of each land-use class, 
and per each cost threshold, is the basis for many subsequent assessment steps.  

2. Aggregate or average total supply

Aggregated class total supply is the number of opportunities (or conversely, total area) 
supplied by all relevant land-use classes within each specified cost threshold. Contrary to that, 
the averaged class total supply is the number of opportunities (or similarly to before, total area) 
averaged across all cost thresholds.

Indicators of diversity of recreational opportunities
-----------------------------------------------------

1. Estimate class diversity

Class diversity indicates the number of distinct land-use classes with assumed recreational 
potential within each cost threshold. 

2. Average class diversity

The average diversity estimates the mean class diversity across cost thresholds.

Indicators of demand and flow
-----------------------------
...

Indicators or cost and proximity
--------------------------------

1. Compute distance (proximity) rasters

For each land-use class of interest, determine distance (proximity) rasters.

2. Determine cost to closest opportunity

Based on the distance (proximity) rasters, determine for each land-use class of interest the
cost to the closest opportunity, and average this cost for all opportunities within reach.  