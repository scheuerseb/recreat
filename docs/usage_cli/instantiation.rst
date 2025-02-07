Instantiate a new model
=======================

At the CLI, recreat_util is the tool used to run a recreat model. As part of calling recreat_util, 
a new recreat model will be instantiated automatically. Therefore, no specific instantiation is 
required at the command-line. 

However, for each recreat model, a data-path needs to be set:

-w, --data-path       Set path to data-path folder.

Example:

.. code-block::
   
   recreat_util -w /path/to/data [...]

.. note::

   If not explicitly specified, the current directory will be used as data-path.

.. note::

   Upon the instantiation of a new recreat model, a set of folders will be created in the model's 
   ``root-path``. Subsequently, outputs will be written into these folders.
