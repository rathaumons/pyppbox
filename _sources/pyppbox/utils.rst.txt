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

Ground-truth files and video-to-GT maps use UTF-8, with or without a UTF-8 BOM.
Non-Latin identities and filenames are preserved. Convert older files saved
in a local encoding such as CP1252 to UTF-8 before loading them. These readers
are also used by online and offline evaluation.

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

``ResIO.dump()`` and ``dumpAll()`` write UTF-8 files named
``res_<timestamp>_full.txt``. If that name is taken, exports use
``res_<timestamp>_full_1.txt``, ``res_<timestamp>_full_2.txt``, and so on.
Exclusive filename reservation protects earlier results during rapid or
concurrent exports. Failed writes clean up their temporary file and reservation.

Both methods return ``None`` and log the actual output path. Stored records
remain available after export. Choose an existing writable output directory;
a missing directory falls back to the package's ``data/res`` directory.

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

