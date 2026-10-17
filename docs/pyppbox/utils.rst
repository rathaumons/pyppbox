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

Logging and writable runtime state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pyppbox uses the named ``pyppbox`` logger and leaves the application's root
logger, handlers, and levels unchanged. Application logging handlers can receive
its records even when pyppbox file logging is disabled. Terminal helpers retain
their existing printed messages and per-call overrides. ``disable_terminal_log()``
controls those prints; host-installed logging handlers remain under application
control. A null handler prevents unsolicited logging output in an unconfigured host.

File logging remains disabled by default. Set ``PYPPBOX_DISABLE_FILE_LOG=0`` to
enable it. The first message opens a unique UTF-8 file under the user runtime
directory's ``logs`` subdirectory, using a timestamp, process ID, and random token.
Unrelated application/library records are not written into that file. Importing
logging utilities creates no runtime files, even when file logging is enabled.
``set_file_log_from_env()`` refreshes the toggle; disabling closes only pyppbox's
owned file handlers. File-creation/write failures disable that sink and report one
warning through the remaining outputs; processing continues. Explicit host choices
for the ``pyppbox`` logger's level and propagation are respected.

``PYPPBOX_RUNTIME_DIR`` overrides the entire runtime directory. Otherwise it is:

* Windows: ``%LOCALAPPDATA%/pyppbox`` (or ``~/AppData/Local/pyppbox``).
* macOS: ``~/Library/Application Support/pyppbox``.
* Other systems: ``$XDG_STATE_HOME/pyppbox`` (or ``~/.local/state/pyppbox``).

GUI launches get a separate temporary handoff directory under ``sessions``.
The launcher and demo inherit ``PYPPBOX_GUI_SESSION_DIR``; concurrent launches do
not share their active config-selection/video state files. Remembered preferences
live under ``gui`` and seed each new launch once. These files use the existing
plain-text format, atomic replacement, and UTF-8. Missing or malformed files have
safe defaults. Legacy installed ``gui/tmp/ui.tmp`` and ``input.tmp`` remain read-only
fallbacks when no remembered state exists. Normal launcher exit or process-start
failure removes owned session files. An abrupt process/system crash can leave a
session directory behind; no automatic deletion of another launch's directory occurs.

Runtime relocation leaves configuration/model/dataset paths and result-export
destinations unchanged. GUI configuration edits still write the selected YAML
directory. Use a writable custom config directory when editing a read-only
installation. Two launchers deliberately selecting the same config directory still
share those configuration files; use separate directories for independent edits.
Remembered preferences use the last successful write, while active sessions stay
independent. Third-party backends may manage their own logging and runtime files.

GUI launches own their launcher/demo processes and workers. Completion or
interruption stops owned processes; normal API returns also clean launch state.
Windows uses private kill-on-close jobs; POSIX uses separate sessions with
parent-loss monitoring. Abrupt process termination can leave abandoned session
folders, as described above; it does not make those folders shared config stores.

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

