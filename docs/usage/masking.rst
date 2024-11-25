
Mask land-uses
==============

Land-use masking, i.e., the generation of  presence-absence-masks for each relevant land-use class, is a key step in each recreat model. 
Similar to clumps, these land-use masks (or class masks, respectively) are a key input for subsequent recreat methods. 

.. note::
    
    For each class included in the model as patch class or edge class (including buffered-edge classes), a corresponding class 
    mask will be written into the MASKS folder.  

Depending on whether recreat is used from the CLI through recreat_util, or in a script, land-use masks are created as follows.

Using recreat_util from the CLI
-------------------------------

When using recreat_util, land-use masks are determined through the ``mask-landuses`` subcommand. This subcommand has no arguments or options.

Example:

.. code-block::

    recreat_util [...] mask-landuses [...]

As part of calling recreat using recreat_util, in the example, land-use masks are determined through the ``mask-landuses`` subcommand. 

In a script 
-----------

In a script, land-use masking is conducted through the :py:meth:`.mask_landuses` method. 

.. code-block:: python

    my_model.mask_landuses()

In the example, land-use masks will be determined through the :py:meth:`.mask_landuses` method.