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

Lifecycle and safe reconfiguration
----------------------------------

Use one pipeline per independent stream, and close it when finished::

    from pyppbox.standalone import MT

    with MT() as pipeline:
        pipeline.setMainModules({
            "detector": "None", "tracker": "Centroid", "reider": "None"
        })
        # Detect, track, and identify this stream's frames here.

``MT.close()`` releases owned models, tracking metadata, and TensorFlow sessions.
It is safe to repeat after successful cleanup. Processing or reconfiguration on a
closed MT raises ``RuntimeError``; create a new instance. The pipeline does not close
caller-owned videos, windows, recorders, or Person objects, and does not clear global
CUDA allocator caches. Allocator memory may remain reserved by the backend.

``MT.reset(frame_index=0)`` clears tracker histories, identity caches, fallback-ID
position, and ReID scratch state while retaining detector and ReID models and
classifiers. It also sets the GT detector's next frame cursor. It does not change
existing Person objects or an explicit full-GT override. SORT's process-wide ID
counter is not reset, so IDs need not restart at zero. Call between streams; calls
on the same MT are not synchronized for concurrent processing/reset/reconfiguration.

Stage setters construct replacements before publishing them. Parsing, validation,
or initialization failure preserves the active stage and configurator. An all-stage
load publishes its configuration and stages only after every candidate succeeds;
earlier candidates are closed if a later stage fails. Configuration-only loads
retain active instances. The public configurator object's identity is preserved.
Invalid or unsupported stage selections now raise ``ValueError`` while retaining the
working stage. Successful replacement closes the previous owned stage.

This policy temporarily holds old and new models together. If memory is insufficient,
initialization fails and the old pipeline remains available. To intentionally release
memory first, call ``close()`` and then construct a new pipeline; that explicit choice
gives up recovery of the old model. No new configuration keys or path rules are needed.

Cleanup attempts every stage even if one close fails. Failed cleanup remains owned
for a later ``close()`` retry. A successful replacement remains active if retiring its
old backend fails; the failure is logged. Explicit close reports unresolved cleanup
as ``RuntimeError``. A context manager preserves an exception from its body rather
than masking it with a cleanup error.

The shared facade exposes ``reset()`` and ``close()`` too. Neither initializes an
unused pipeline. After a successful facade ``close()``, the next ordinary facade
call creates a fresh shared instance. Cleanup failure retains the closed instance
for another close attempt. Use these operations without concurrent facade calls.

Classifier training constructs one temporary reider and publishes it on success.
Training failure preserves the active reider. Output classifier files are still
written by the existing training code; they are not part of a pipeline-state
transaction. FaceNet training sessions are closed on completion or failure, and
failed classifier reloads retain the prior working classifier/session.

pyppbox.standalone
------------------

:py:mod:`pyppbox.standalone`

.. automodule:: pyppbox.standalone
   :members: setConfigDir, setMainModules, getConfig, getMainConfig, forceFullGTMode, setMainDetector, detectPeople, setMainTracker, trackPeople, setMainReIDer, reidPeople, trainReIDClassifier, reset, close
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
