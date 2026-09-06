.. _config-page:

Configurations
==============

Configuration inputs and paths
------------------------------

V3.15 keeps the existing configuration keys and file layout. Module setters accept
a supported name, a dictionary, a raw YAML/JSON mapping, or a YAML/JSON file path.
A legacy list containing one mapping is also accepted. Single-module setters need
one document; files listing several modules belong in a configuration directory
or in the configurator's bulk loaders.

The internal configuration directory contains ``main.yaml``, ``detectors.yaml``,
``trackers.yaml``, and ``reiders.yaml``. Paths inside these internal configs resolve
relative to the installed ``pyppbox`` directory. Paths in custom configurations
resolve relative to the current working directory unless a config class is created
with ``relative_to_pyppbox_root=True``. Absolute model paths work in either case.
The location of a custom YAML/JSON file does not become the base for its model paths.

Use ``setConfigDir(config_dir=None, load_all=True)`` to load the selected internal
modules. The default ``load_all=False`` selects the config directory without loading
models. The module name ``"None"`` disables a stage; it is a string.

Use native booleans for options such as ``show_boxes``. V3.15 also accepts existing
quoted ``"True"``/``"False"`` values for that field. Config saves preserve general
string values and replace each file atomically; saving several files is not a single
transaction.

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
