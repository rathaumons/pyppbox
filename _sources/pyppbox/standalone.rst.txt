.. _standalone-page:

Standalone Module
=================

The functions in ``pyppbox.standalone`` share one pipeline instance. Use them for
a single stream. For independent streams, create one :class:`pyppbox.ppb.mt.MT`
instance per stream. Both ``pyppbox.standalone`` and ``pyppbox.ppb`` remain supported.

pyppbox.standalone
------------------

:py:mod:`pyppbox.standalone`

.. automodule:: pyppbox.standalone
   :members: setConfigDir, setMainModules, getConfig, getMainConfig, forceFullGTMode, setMainDetector, detectPeople, setMainTracker, trackPeople, setMainReIDer, reidPeople, trainReIDClassifier
   :undoc-members: MT
   :show-inheritance:

|

pyppbox.standalone.MT
---------------------

:py:class:`pyppbox.ppb.mt.MT`

.. automodule:: pyppbox.ppb.mt
   :members:
   :undoc-members:
   :show-inheritance:

|
