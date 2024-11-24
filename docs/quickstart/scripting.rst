Using recreat from scripts
==========================

Simply import recreat using the import statement:

.. code-block:: python

   from recreat import recreat
   

Define a data path, which includes your data and into which results will be written, and instantiate the class:

.. code-block:: python

   # set to your data path
   my_data_path = "/path/to/data/"
   my_model = recreat(my_data_path)

Subsequently, you can define the parameters in your model using the set_params method:

.. code-block:: python

   # define relevant land-use classes and costs
   my_model.set_params('classes.patch', [24, 25, 26]) 
   my_model.set_params('classes.edge', [39, 44])
   my_model.set_params('classes.builtup', [1, 2])
   my_model.set_params('costs', [21, 101])

Import the land-use raster for a given scenario, here, current, using the set_land_use_map method; define values to be treated as no-data value, and values to replace these values, as needed, for example:

.. code-block:: python

   # import land-use raster for current scenario
   my_model.set_land_use_map('current', 'U2018_CLC2018_V2020_20u1.tif', nodata_values=[-128.0], nodata_fill_value = 0)


Then, invoke actions through calling recreat's methods, as desired. For example, to detect clumps, mask land-uses, detect edges, and determine supply of each land-use class within cost thresholds:

.. code-block:: python

   # do something
   my_model.detect_clumps(barrier_classes=[39, 44])
   my_model.mask_landuses()   
   my_model.detect_edges(grow_edges=[39, 44])
   my_model.class_total_supply()

Please refer to the usage section for more information.