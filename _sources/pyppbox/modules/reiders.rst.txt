Supported ReIDers
=================

FaceNet | ``MyFaceNet``
-----------------------

TensorFlow-based face recognition. See :ref:`device-selection` for device behavior.

.. automodule:: pyppbox.modules.reiders.facenet
   :members:
   :undoc-members:
   :show-inheritance:

Torchreid | ``MyTorchreid``
---------------------------

PyTorch-based body recognition, with optional batched inference.

.. automodule:: pyppbox.modules.reiders.torchreid
   :members:
   :undoc-members:
   :show-inheritance:

ArcFace and AdaFace
-------------------

ArcFace R50 and AdaFace IR18 use ONNX Runtime with SCRFD face detection and
112x112 alignment. Install the optional runtime and the selected model/data
wheel from :ref:`getstarted-page`. Each wheel supplies its own detector,
embedding model, and GTA V identity classifier. CPU is the default;
see :ref:`device-selection` for CUDA selection.

Select either reider in the GUI or pipeline::

    from pyppbox.ppb.mt import MT

    with MT() as pipeline:
        pipeline.setMainReIDer('ArcFace')  # or 'AdaFace'
        # Pass detected/tracked people to pipeline.reidPeople().

For one face image::

    from pyppbox.config.myconfig import RCFGArcFace
    from pyppbox.modules.reiders.onnxface import MyArcFace

    backend = MyArcFace(RCFGArcFace(relative_to_pyppbox_root=True), auto_load=True)
    try:
        identity, confidence = backend.recognize_file('face.jpg')
    finally:
        backend.close()

Use ``RCFGAdaFace`` and ``MyAdaFace`` for AdaFace. ``extract_embedding(image)``
returns a normalized embedding or ``None`` when no face is detected.
``embed_aligned(image)`` accepts a BGR 112x112 image and skips detection.

Training identities
~~~~~~~~~~~~~~~~~~~

Put JPG/PNG training images in one subdirectory per identity, then call
``trainReIDClassifier()`` with the selected reider, training directory, and
output classifier path. This trains an SVM while keeping the embedding model
frozen. See :doc:`../../examples/example_12` and
:doc:`../../examples/example_18` for complete examples.

Classifiers are specific to their backend, model, and preprocessing settings.
Incompatible classifiers are rejected. Load only trusted pickle files.
Confidence is an SVM probability on a 0–100 scale; ``min_confidence`` uses 0–1.

The SCRFD adapter includes MIT-licensed InsightFace code; see ``NOTICE`` and
``licenses/InsightFace-MIT.txt``. The InsightFace Python package is not required.
Model weights have separate terms; see the notices in the selected data wheel.

.. automodule:: pyppbox.modules.reiders.onnxface
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
