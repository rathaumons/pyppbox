Supported Trackers
==================

Config Name | ``Class Name``

Centroid, SORT, and DeepSORT use ``laprust>=1.0.0`` for assignment solving.
Centroid and DeepSORT retain their SciPy fallback if laprust is unavailable or
the solve fails. Matching thresholds are unchanged, but equally optimal
assignments can differ from those produced by the previous lapx solver.

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
