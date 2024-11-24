Prerequisites
=============

recreat requires gridded land-use data, and for determining demand and flows, 
an additional gridded population dataset. These datasets should be located in the 
so-called ``root-path`` of the model. The ``root-path`` may be considered as a given scenario, 
such as current situation or future land-use and population projections, 
that shall be assessed. The ``root-path``, in turn, should be located 
within the ``data-path`` of recreat. 

.. code-block::

   data-path
   ├── root-path1
   |   ├── land-use raster
   |   └── (population raster)
   ├── root-path2
       ├── land-use raster
       └── (population raster)
   ...