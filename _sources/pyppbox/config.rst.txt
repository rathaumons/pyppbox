.. _config-page:

Configurations
==============

Inputs and paths
----------------

Module setters accept a module name, dictionary, YAML/JSON mapping, or file
path. Multi-document files belong in a config directory or bulk loader.
``"None"`` disables a stage.

A config directory contains ``main.yaml``, ``detectors.yaml``, ``trackers.yaml``,
and ``reiders.yaml``. Internal model paths resolve from the installed package;
custom paths resolve from the working directory. Absolute paths work in both.
A custom YAML file's location does not change this base directory.

``setConfigDir(None, load_all=True)`` loads the selected internal modules.
The default ``load_all=False`` changes the configuration without loading models.
Use ``relative_to_pyppbox_root=True`` when constructing config objects that
should resolve paths from the installed package.

.. _device-selection:

Devices
-------

Select a preset or type a supported value in the GUI's device field:

* **RF-DETR:** GPU index, ``cpu``, ``cuda:N``, ``mps``, or ``auto``; default ``0``.
* **YOLO Ultralytics:** backend-supported device or GPU indices; an empty value
  selects automatically.
* **Torchreid:** PyTorch device, such as ``cpu``, ``cuda``, or ``cuda:N``.
* **ArcFace/AdaFace:** ``cpu``, ``cuda``, or ``cuda:N``; default ``cpu``.
* **YOLO Classic:** OpenCV selects CPU/CUDA according to its build and hardware.
* **FaceNet/DeepSORT:** TensorFlow selects devices; FaceNet's ``gpu_mem`` limits
  GPU memory and does not select a device.

Device selection requires a compatible installed runtime. Opening a settings
panel does not load a model. See :doc:`../getstarted` for dependencies.

Validation and saving
---------------------

Settings are validated before use, and inference files are checked before
loading models. Training data and GT mapping files are not required for normal
inference. RF-DETR, ArcFace, and AdaFace reject unknown configuration keys.

GUI saves and ``dumpDetectorCFG()``, ``dumpTrackerCFG()``, and ``dumpReiderCFG()``
merge edited fields, preserving other modules and extension fields. The bulk
``dumpAll*CFG()`` methods replace complete document lists. File replacement is
atomic; YAML comments are regenerated.

pyppbox.gui.guitools
--------------------

.. automodule:: pyppbox.gui.guitools
   :members:
   :undoc-members:
   :show-inheritance:

pyppbox.config.configtools
--------------------------

.. automodule:: pyppbox.config.configtools
   :members:
   :undoc-members:
   :show-inheritance:

pyppbox.config.myconfig
-----------------------

.. automodule:: pyppbox.config.myconfig
   :members:
   :undoc-members:
   :show-inheritance:

pyppbox.config.unifiedstrings
-----------------------------

.. automodule:: pyppbox.config.unifiedstrings
   :members:
   :undoc-members:
   :show-inheritance:

pyppbox.config.torchreidmodels
------------------------------

.. automodule:: pyppbox.config.torchreidmodels
   :members:
   :undoc-members:
   :show-inheritance:
