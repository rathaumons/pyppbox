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

Opt-in Torchreid inference batches
----------------------------------

``reidPeople()`` retains scalar inference by default. For a Torchreid pipeline,
enable bounded batches per call::

    people, counts = pipeline.reidPeople(
        frame, people, img_is_mat=True, batch_size=8
    )

The same keyword works through ``pyppbox.standalone.reidPeople()``. It is a positive
integer maximum per model/classifier call, not a minimum or an across-frame queue.
Only the current chunk's crops are prepared; no frames are delayed or dropped.
No YAML changes are required. Existing configurations, scalar ``recognize()``, and
FaceNet inference/training settings retain their behavior. The option applies to
Torchreid; other reiders continue their existing scalar processing.

Known identities are retained during the normal pass. Duplicate groups are processed
in their original order, including identities moved into a later group by an earlier
recognition. ReID counts remain completed person recognitions for the initial and
duplicate passes, including unknown/error-ID results. Batch calls, numerical decision
rechecks, and retries do not inflate those counts. Invalid crops or failed scalar
recognition do not count. A failed batch retries its people individually. Repeated
references to one Person object use the scalar path to retain mutation order.

Batch floating-point arithmetic can slightly change confidence values. Results near
the configured threshold or with nearly tied winning classes are rechecked using
scalar inference (0.1 percentage points around the threshold; 0.2 points between
the top two classes). These guards protect nearby decisions but are not a universal
error bound for custom models/devices. Validate identity, confidence, and count
equivalence for your models before enabling batches. Keep ``batch_size=1`` when
requiring the original scalar execution. Speed and temporary memory depend on the
device, model, person count, and selected limit; larger batches are not always faster.

Direct reider users can call ``MyTorchreid.recognize_batch(images, batch_size=8)``.
It accepts an iterable, processes bounded chunks, and returns identity/confidence
pairs in input order. Confidence stays on the 0-100 scale. Unlike the pipeline,
the direct method propagates errors without returning a partial result. It retains
the complete returned result list; its temporary inference inputs are chunk-bounded.

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
