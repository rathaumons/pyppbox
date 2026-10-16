.. _standalone-page:

Standalone Module
=================

The functions in ``pyppbox.standalone`` share one pipeline instance. Use them for
a single stream. For independent streams, create one :class:`pyppbox.ppb.mt.MT`
instance per stream. Both ``pyppbox.standalone`` and ``pyppbox.ppb`` remain supported.

Importing the facade functions defers pipeline imports and shared-instance
construction until the first function call. That first call pays the deferred
setup cost. Importing ``MT`` explicitly loads the class for direct construction;
each ``MT()`` instance remains independent of the shared facade state.
Wildcard imports include ``MT`` and therefore also load the pipeline class.

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
