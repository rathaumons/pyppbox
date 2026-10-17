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
string values and replace each file atomically.

Validation and safe saves
-------------------------

The V4 development changes retain these formats, keys, defaults, and path rules.
Supported config classes validate required settings, finite numeric values, positive
image/batch sizes, nonnegative tracker counts, path types, and device syntax before
initializing models. Numeric strings and quoted boolean values are normalized where
safe. Signed representative-point and face-crop offsets, zero SORT counts, and
``nn_budget=None`` remain valid. DeepSORT cosine distance may exceed one.

Reading a configuration or opening its settings panel does not require installed
model files. The pipeline checks inference assets before constructing a stage and
checks deferred ReID classifier assets when loading them. Training directories and
GT mapping files are not required for ordinary inference. Ultralytics retains its
backend-managed downloads and supported model formats; RF-DETR retains its explicit
local-checkpoint or empty-path download modes. Device availability and checkpoint
contents remain backend checks. ``BaseCGF.validateRuntime(classifier=True)`` also
provides an explicit preflight for config objects used outside the pipeline.

The main save and targeted ``dumpDetectorCFG()``, ``dumpTrackerCFG()``, and
``dumpReiderCFG()`` methods merge edited fields, retaining extension fields and the
order of other documents. Settings panels use these targeted saves and remain open
if validation or writing fails. Unknown module documents remain opaque. RF-DETR's
existing rejection of unknown options is unchanged. YAML comments are regenerated.
The bulk ``dumpAll*CFG()`` methods still replace complete document lists.

For several files, ``MyConfigurator.dumpConfigs()`` validates and stages complete
documents before replacing any file. A handled write error restores the exact
original bytes. If the process is interrupted, retained ``.pyppbox-save-*`` folders
allow explicit recovery::

    from pyppbox.config.configtools import recoverConfigSave
    recoverConfigSave("my-config-directory")

Stop users of that config directory before recovery. Use one writer and pause
readers during a group save: readers can observe intermediate file combinations.
Recovery restores unfinished saves and retains completed saves. Backups remain
available if restoration fails, and configurator saves refuse to overwrite a pending
group. Resolved symlink targets must share the config directory's filesystem.
Power-loss durability depends on the filesystem. Individual file replacement
remains atomic; this does not provide concurrent multi-file transactions.

Torchreid model metadata
------------------------

Configuration loading and the Torchreid settings panel read the model catalog
through ``pyppbox.config.torchreidmodels`` without importing the ReID runtime
package. The catalog YAML file, custom catalog paths, filename matching, and
image dimensions retain their existing behavior. Imports from
``pyppbox.modules.reiders.torchreid.model_dict`` remain supported.

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
