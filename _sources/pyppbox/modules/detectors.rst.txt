Supported Detectors
===================

Config Name | ``Class Name``

----

YOLO_Classic | ``MyYOLOCLS``
----------------------------

The popup's last parameter row shows ``device`` and ``auto by OpenCV`` before
Save. The adapter selects
OpenCV's CUDA DNN backend when OpenCV reports a CUDA-enabled device; otherwise
it uses OpenCV's default CPU backend. There is no ``device`` configuration key.

.. automodule:: pyppbox.modules.detectors.yolocls
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

YOLO_Ultralytics | ``MyYOLOULT``
--------------------------------

The device field is an editable dropdown with ``0``, ``cpu``, ``cuda:0``, and
``mps`` presets. Other supported values can be typed, including GPU indices and
``cuda:N``. Clear the field for automatic selection; the literal ``auto`` is not
a YOLO device value in pyppbox. See :ref:`device-selection` for normalization,
availability, and defaults.

.. automodule:: pyppbox.modules.detectors.yoloult
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

RF_DETR | ``MyRFDETR``
----------------------

Optional RF-DETR detection with Nano, Small, Medium, and Large models. See
:doc:`../../examples/example_15` for installation, native class IDs, checkpoints,
GUI settings, and a complete tracking example. Configuration uses
:class:`pyppbox.config.myconfig.DCFGRFDETR`.

The editable device dropdown offers ``0``, ``cpu``, ``cuda:0``, ``mps``, and
``auto``. Type ``cuda:N`` for another GPU, or use ``auto`` to delegate selection
to RF-DETR. The configured default remains GPU index ``0``. See
:ref:`device-selection` for the differences from other modules.

.. automodule:: pyppbox.modules.detectors.rfdetr
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
