Example 3: Set a custom config directory
========================================

- **Description:** Use a custom config directory to set the configurations for the supported modules.
- **Featuring:**
   - :py:func:`~pyppbox.standalone.setConfigDir`
   - :py:func:`~pyppbox.standalone.detectPeople`
   - :py:func:`~pyppbox.standalone.trackPeople`
   - :py:func:`~pyppbox.standalone.reidPeople`
   - :py:func:`~pyppbox.utils.visualizetools.visualizePeople`

ℹ️ **Source code and input file(s)** -> `{pyppbox repo}/examples`_

.. _{pyppbox repo}/examples: https://github.com/rathaumons/pyppbox/tree/dev/examples

The ``examples/cfg/`` directory contains configurations for every supported
detector, tracker, and reider, including ``RF_DETR``, ``ArcFace``, and ``AdaFace``.
Replace ``C:/pyppbox_v3`` in the module config files with your installed
``pyppbox`` directory. Select the stages in ``cfg/main.yaml``; for example:

.. code-block:: yaml

   detector: RF_DETR
   tracker: SORT
   reider: ArcFace

Use ``reider: AdaFace`` to select AdaFace instead. The new modules use ``cpu``
in these sample configs. Install ``.[rfdetr]`` and/or ``.[faces]`` from the
repository root and obtain the data wheels for the selected modules before
running. ArcFace and AdaFace each need their own model and classifier paths.
See :doc:`../pyppbox/modules/detectors`, :doc:`../pyppbox/modules/reiders`, and
:ref:`device-selection` for requirements and device choices.

Example source
--------------

.. literalinclude:: ../../examples/example_03_custom_cfg_directory.py
   :encoding: utf-8
   :language: python
