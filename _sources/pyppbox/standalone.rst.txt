.. _standalone-page:

Standalone Module
=================

Use ``pyppbox.standalone`` for a shared pipeline, or create an ``MT`` instance
for each independent stream. The functions below share the same interfaces.
See :doc:`../examples` for complete detection, tracking, and ReID examples.

Lifecycle
---------

Use a context manager to release models and sessions::

    from pyppbox.ppb.mt import MT

    with MT() as pipeline:
        pipeline.setMainDetector('YOLO_Ultralytics')
        people, image = pipeline.detectPeople('image.jpg')

``reset()`` clears stream state while keeping loaded models. ``close()``
releases resources; a closed ``MT`` cannot process more frames. Closing the
shared standalone pipeline allows its next call to create a fresh instance.
Use separate instances for concurrent streams.

Model replacement loads the new backend before releasing the old one, so a
failed load leaves the current pipeline usable. Allow memory for both during
replacement. ``close()`` can retry a failed cleanup.

Torchreid batches
-----------------

Torchreid can process crops in batches::

    people, count = pipeline.reidPeople(frame, people, img_is_mat=True, batch_size=8)

The default ``batch_size=1`` preserves scalar inference. Batching can change
confidence slightly and is not always faster. Direct backend users can call
``MyTorchreid.recognize_batch()``; see its API below for result and error behavior.

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
