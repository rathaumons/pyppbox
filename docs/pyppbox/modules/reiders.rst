Supported ReIDers
=================

Config Name | ``Class Name``

----

FaceNet | ``MyFaceNet``
-----------------------

.. automodule:: pyppbox.modules.reiders.facenet
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

Torchreid | ``MyTorchreid``
---------------------------

.. automodule:: pyppbox.modules.reiders.torchreid
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

ArcFace and AdaFace
-------------------

These optional face recognizers use SCRFD detection, five-point alignment, a
local ONNX embedding model, and a separately trained identity SVM. They use the
existing ``Person.faceid`` / ``faceid_conf`` fields, crop calibration, duplicate
pass, configuration APIs, and GUI settings. FaceNet retains its existing behavior.

Install the optional dependencies from this ``dev`` checkout with Python 3.10
or newer::

   python -m pip install ".[faces]"

Packaged models and GTA V classifiers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install either or both v1.6.0 data wheels::

   python -m pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.6.0/pyppbox_data_arcface-1.6.0-py3-none-any.whl
   python -m pip install https://github.com/rathaumons/pyppbox-data/releases/download/v1.6.0/pyppbox_data_adaface-1.6.0-py3-none-any.whl

Each contains its own SCRFD detector, ONNX embedding model, and model-specific
``classifier/gta5.pkl``. AdaFace does not require the ArcFace wheel. The installed
paths match the internal defaults; select a backend by name::

   from pyppbox.ppb.mt import MT

   with MT() as pipeline:
       pipeline.setMainReIDer('ArcFace')  # or 'AdaFace'; CPU by default
       # Pass detector/tracker Person objects to pipeline.reidPeople().

For a cropped face image, use the public backend directly::

   from pyppbox.config.myconfig import RCFGArcFace
   from pyppbox.modules.reiders.onnxface import MyArcFace

   config = RCFGArcFace(relative_to_pyppbox_root=True)
   backend = MyArcFace(config, auto_load=True)
   try:
       identity, confidence_percent = backend.recognize_file('/path/to/face.jpg')
   finally:
       backend.close()

Use ``RCFGAdaFace`` and ``MyAdaFace`` for the other model. The packaged classifiers
identify Amanda, Franklin, Lester, Michael, and Trevor from GTA V. FaceNet used
2,957 training faces; ArcFace and AdaFace each used 2,949 and skipped eight images
with no detected face. The two Torchreid classifiers each used 150 body crops.
These full-dataset training counts are not held-out recognition scores.

All five identity SVMs were trained with scikit-learn 1.9.0 on Python 3.11.16.
These versions describe the training environment. Training also used
NumPy 2.4.6, SciPy 1.17.1, scikit-image 0.26.0, TensorFlow 2.21.0, and
PyTorch 2.5.1+cu121; the ONNX backends used ONNX Runtime GPU 1.20.2 and InsightFace
2.1. The data wheels install assets only. Per-module ``TRAINING.json`` and
``WEIGHTS.json`` record training provenance and file hashes.

Device selection and model provenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The base install provides CPU ONNX Runtime. For NVIDIA inference, use a compatible
``onnxruntime-gpu`` distribution instead of ``onnxruntime``. Match its CUDA and
cuDNN requirements to the installed runtime. Requested CUDA initialization errors
raise instead of silently producing CPU results. See the
`ONNX Runtime CUDA compatibility guide <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>`_.

Weights are explicit local files; configuration inspection never downloads or
loads a model. For ArcFace, select the detector ``det_10g.onnx`` and embedding
``w600k_r50.onnx`` from InsightFace's official ``buffalo_l`` pack. Its supplied
weights are designated for noncommercial research; consult the upstream terms.

AdaFace defaults match an ONNX export of the author's
`CVLFace IR18 VGG2 model <https://huggingface.co/minchul/cvlface_adaface_ir18_vgg2>`_.
``examples/example_16_export_adaface_onnx.py`` exports those weights through the
original AdaFace network and verifies the ONNX output against PyTorch. Other
AdaFace exports may require different preprocessing; the original AdaFace
inference code uses BGR, whereas this CVLFace checkpoint uses RGB. Match
``input_color``, ``input_mean``, and ``input_std`` to the checkpoint.

Training your own identities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure and train a new identity classifier::

   from pyppbox.ppb.mt import MT

   cfg = {
       'ri_name': 'ArcFace',  # or AdaFace
       'model_det': '/models/det_10g.onnx',
       'model_file': '/models/w600k_r50.onnx',
       'classifier_pkl': '/classifiers/arcface.pkl',
       'train_data': '/datasets/my_faces',
       'device': 'cpu',
   }
   with MT() as pipeline:
       pipeline.trainReIDClassifier(cfg)
       # On the next reidPeople() call, the new classifier is loaded lazily.

Training folders contain image files in one subdirectory per identity. Training
detects and aligns faces, reports skipped undetected images, and rejects any
identity with no detectable samples. The output classifier directory must exist.
This trains the SVM, not the embedding network. For direct recognition, see
``examples/example_18_onnx_face_reid.py``; the existing pipeline also accepts
``setMainReIDer(cfg)`` or a configured ``'ArcFace'`` / ``'AdaFace'`` selection.

Each classifier stores the backend, checkpoint hashes, and preprocessing
signature. FaceNet classifiers and classifiers from different model exports are
rejected. Use only trusted pickle files. ``min_confidence`` is an SVM probability
threshold, not a cosine threshold or a calibrated unknown-person probability.
Validate unknown identities and target capture conditions before choosing it.

``extract_embedding(image)`` returns a normalized feature vector, or ``None`` if
no face is detected. ``embed_aligned(image)`` accepts a BGR 112x112 aligned image
and skips detection. These methods allow separate embedding and complete-pipeline
benchmarks. Inference is scalar; the pipeline's Torchreid batching option retains
its existing scope. ``examples/example_17_benchmark_faces.py`` records versions,
latency, detection failures, held-out identity classification, and cosine-pair
metrics. Its file split does not establish independence between capture sessions.

.. automodule:: pyppbox.modules.reiders.onnxface
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
