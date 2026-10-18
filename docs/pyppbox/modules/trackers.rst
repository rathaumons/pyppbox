Supported Trackers
==================

Config Name | ``Class Name``

----

Centroid | ``MyCentroid``
-------------------------

.. automodule:: pyppbox.modules.trackers.centroid
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

SORT | ``MySORT``
-----------------

.. automodule:: pyppbox.modules.trackers.sort
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

DeepSORT | ``MyDeepSORT``
-------------------------

The popup's last parameter row shows ``device`` and ``auto by TensorFlow`` before
Save. TensorFlow selects the appearance encoder's device automatically, according
to the installed runtime and available devices.
There is no ``device`` configuration key. Tracking and assignment run on CPU.
See :ref:`device-selection` for the distinction from editable device dropdowns.

.. automodule:: pyppbox.modules.trackers.deepsort
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
