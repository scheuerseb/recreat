Instantiate a new model
=======================

Depending on whether recreat is used from the CLI through recreat_util, or in a script, 
a new recreat model is instantated as follows.

Using recreat_util from the CLI
-------------------------------

At the CLI, recreat_util is the tool used to run a recreat model. As part of calling recreat_util, 
a new recreat model will be instantiated automatically. Therefore, no specific instantiation is 
required at the command-line. 

However, for each recreat model, a data-path needs to be set. At the CLI, 
this data-path may be specified through the ``-w``/``--data-path`` option:

.. code-block::
   
   recreat_util -w /path/to/data [...]

.. hint::

   If not explicitly specified, the current directory will be used as data-path.


In a script
-----------

In a script, import the recreat package and create a new recreat model 
through instantiating the recreat class as follows:

.. code-block:: python

   from recreat import recreat

   # define the data-path
   data_path = "C:/Users/sebsc/Desktop/CLC"

   # instantiate recreat model, and set data-path
   my_model = recreat(data_path=data_path)

As shown in the example, initializing the recreat class requires setting the ``data-path``, 
with the data_path being the only argument to the class constructor.

.. hint::

   Upon the instantiation of a new recreat model, a set of folders will be created in the model's 
   ``root-path``. Subsequently, outputs will be written into these folders.
