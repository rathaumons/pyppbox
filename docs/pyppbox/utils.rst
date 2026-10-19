.. _utils-page:

Utilities
=========

pyppbox.utils.evatools
----------------------

.. automodule:: pyppbox.utils.evatools
   :members:
   :undoc-members:
   :show-inheritance:

pyppbox.utils.gttools
---------------------

.. automodule:: pyppbox.utils.gttools
   :members:
   :undoc-members:
   :show-inheritance:

pyppbox.utils.logtools
----------------------

Logging uses the named ``pyppbox`` logger without configuring the application's
root logger. File logging is off by default; set ``PYPPBOX_DISABLE_FILE_LOG=0``
to enable it, or use the functions below. Log I/O failures do not stop inference.

Logs and GUI state live in the user's runtime directory:

* Windows: ``%LOCALAPPDATA%/pyppbox``.
* macOS: ``~/Library/Application Support/pyppbox``.
* Linux: ``$XDG_STATE_HOME/pyppbox`` or ``~/.local/state/pyppbox``.

``PYPPBOX_RUNTIME_DIR`` overrides this location. GUI launches have separate
session state and clean up owned workers on exit. Model, configuration, and
result paths retain their existing locations; use a writable custom config
directory when editing a read-only installation.

.. automodule:: pyppbox.utils.logtools
   :members:
   :undoc-members:
   :show-inheritance:

pyppbox.utils.runtime
---------------------

.. automodule:: pyppbox.utils.runtime
   :members:

pyppbox.utils.mot2pyppbox
-------------------------

.. automodule:: pyppbox.utils.mot2pyppbox
   :members:
   :undoc-members:
   :show-inheritance:

pyppbox.utils.persontools
-------------------------

.. automodule:: pyppbox.utils.persontools
   :members:
   :undoc-members:
   :show-inheritance:

pyppbox.utils.restools
----------------------

.. automodule:: pyppbox.utils.restools
   :members:
   :undoc-members:
   :show-inheritance:

pyppbox.utils.visualizetools
----------------------------

.. automodule:: pyppbox.utils.visualizetools
   :members:
   :undoc-members:
   :show-inheritance:

