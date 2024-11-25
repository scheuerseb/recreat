Class diversity
===============

Class diversity corresponds to the distinct, unique land-use classes (opportunity types) 
relevant for recreation within a given cost threshold. Class diversity is determined for each cost threshold.

.. note::

    Outputs are written into the DIVERSITY folder.

Depending on whether recreat is used from the CLI through recreat_util, or in a script, 
class diversity is determined as follows.

Using recreat_util from the CLI
-------------------------------

When using recreat_util, class diversity is determined through the ``class-diversity`` subcommand. 
This subcommand has no arguments or options.

Example:

.. code-block::
    
    recreat_util [...] class-diversity [...]

As part of calling recreat using recreat_util, in this example, class diversity is determined through the ``class-diversity`` subcommand.

In a script
-----------

In a script, class diversity is determined through the :py:meth:`.class_diversity` method. 

Example:

.. code-block:: python
    
    my_model.class_diversity()

In this example, class diversity is determined through the :py:meth:`.class_diversity` method.