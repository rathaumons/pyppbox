Supported Detectors
===================

Config Name | ``Class Name``

----

YOLO_Classic | ``MyYOLOCLS``
----------------------------

.. automodule:: pyppbox.modules.detectors.yolocls
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

YOLO_Ultralytics | ``MyYOLOULT``
--------------------------------

.. automodule:: pyppbox.modules.detectors.yoloult
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

GT | ``GTInterpreter``
----------------------

The ``GT`` detector uses :class:`pyppbox.utils.gttools.GTInterpreter` to read the
configured ground-truth text file. See :doc:`../../examples/example_10` for use
through the public pipeline API.
