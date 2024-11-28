Reclassification
================

Reclassification allows modifying class values in the gridded land-use, i.e., to reclassify land-use classes into new class(es). 

.. note::

    By default, reclassification is conducted in memory by modifying the imported land-use raster.

Depending on whether recreat is used from the CLI through recreat_util, or in a script, reclassification is conducted as follows.

Using recreat_util from the CLI
-------------------------------

When using recreat_util, reclassification is conducted using the ``reclassify`` subcommand. This subcommand takes two arguments, first, 
source-classes to be reclassified, and destination-class, i.e., the new class value for al source classes. 

.. code-block::

    recreat_util [...] reclassify [options] <source-classes> <destination-class> [...]

Options:

-e, --export         Export recategorized raster to specified filename (by default, None).  

Example:

.. code-block::

    recreat_util [...] reclassify 1,2 1 [...]

As part of calling recreat using recreat_util, using the reclassify subcommand, land-use classes 1 and 2 are 
reclassified into a new class with the value 1. Multiple reclassify subcommands may be used to reclassify values into multiple 
destination class values.

In a script
-----------

In a script, reclassification is achieved through the :py:meth:`.reclassify` method. The ``mappings`` argument takes a dictionary, where keys correspond 
to the target class, and values to lists with source class values. 

Example:

.. code-block:: python
    
    # define destination and source classes
    new_classes = {1: [1,2]}
    my_model.reclassify(mappings=new_classes)

In the example, land-use classes 1 and 2 (the class values included in the list of source classes as dictionary value) will be reclassified into a new 
class with the value 1 (the dictionary key). Multiple reclassifications may be defined through the dictionary.


