Instantiate a new model
=======================

Import the recreat package and create a new recreat model 
through instantiating the recreat class as follows:

.. code-block:: python

   from recreat import recreat

   # define the data-path
   data_path = "C:/Users/sebsc/Desktop/CLC"

   # instantiate recreat model, and set data-path
   my_model = recreat(data_path=data_path)

As shown in the example, initializing the recreat class requires setting the ``data-path``, 
with the data_path being the only argument to the class constructor.
