Supported ReIDers
=================

Config Name | ``Class Name``

----

FaceNet | ``MyFaceNet``
-----------------------

The popup's last parameter row shows ``device`` and ``auto by TensorFlow`` before
Save. TensorFlow's installed build and available devices determine neural-network placement;
FaceNet exposes no ``device`` key. ``gpu_mem`` sets the session's GPU memory
fraction and does not force CPU or GPU use. See :ref:`device-selection`.

.. automodule:: pyppbox.modules.reiders.facenet
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

Torchreid | ``MyTorchreid``
---------------------------

The device dropdown is editable, with ``cpu``, ``cuda``, ``cuda:0``, and ``mps``
presets. Type another supported PyTorch device or an indexed GPU such as
``cuda:1``. Numeric indices normalize to ``cuda:N``; ``auto`` is not accepted.
The bundled default remains ``cuda``. Availability depends on the installed
PyTorch runtime and hardware. See :ref:`device-selection`.

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

pyppbox includes a small SCRFD inference adapter adapted from InsightFace's
MIT-licensed code; attribution and terms are included in ``NOTICE`` and
``licenses/InsightFace-MIT.txt``. The InsightFace Python package is not a runtime
dependency. The adapter uses the configured ONNX session directly and supports
float32 NCHW SCRFD models with five landmarks (9 or 15 outputs), including the
packaged ``det_10g.onnx``. Model weights retain their own licenses.

pyppbox performs the 112x112 alignment through scikit-image's public
``estimate_transform('similarity', ...)`` API. It retains the canonical ArcFace
landmarks and OpenCV warp used by existing classifiers, and supports
scikit-image 0.20 and newer without calling InsightFace's deprecated alignment
helper. No dependency source patches or warning filters are needed.

With Python 3.10 or newer, install one face requirements file from the
repository root. For CPU inference::

   python -m pip install -r requirements/faces.txt

For NVIDIA GPU inference, use the GPU requirements and follow the runtime setup below::

   python -m pip install -r requirements/faces_gpu.txt

These files supplement the base dependencies. Install pyppbox separately as
described in :ref:`getstarted-page`.

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

Use ``RCFGAdaFace`` and ``MyAdaFace`` for the other model. For packaged model and
classifier details, see `pyppbox-data <https://github.com/rathaumons/pyppbox-data/>`_.

Device selection and model provenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both settings dialogs use an editable device dropdown with ``cpu``, ``cuda``,
and ``cuda:0`` presets. CPU is the default. Type ``cuda:N`` or a GPU index for
another NVIDIA device; indices normalize to ``cuda:N``. These ONNX adapters do
not accept ``mps`` or ``auto``. See :ref:`device-selection` for all module choices.

``requirements/faces.txt`` selects ``onnxruntime``; ``requirements/faces_gpu.txt``
selects ``onnxruntime-gpu``. The requirements do not detect your PyTorch/CUDA version.
Match the runtime's CUDA and cuDNN requirements to the installed libraries.
Requested CUDA initialization errors raise instead of silently producing CPU
results. See the
`ONNX Runtime CUDA compatibility guide <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>`_.

For the Windows PyTorch ``2.5.1+cu121`` setup (CUDA 12.1, cuDNN 9),
``onnxruntime-gpu==1.20.2`` has been verified with both face reiders. Install
the GPU requirements with this version::

   python -m pip install -r requirements/faces_gpu.txt "onnxruntime-gpu==1.20.2"

Run this command with the same Python interpreter that launches pyppbox, then
restart its Python/GUI process. This version is specific to that CUDA setup;
other environments should select a matching build from the compatibility guide.
The PyPI ONNX Runtime GPU packages switched to CUDA 13 by default in 1.27.
Importing a CUDA 12 PyTorch build cannot supply their CUDA 13 libraries.

A fresh installation using ``faces_gpu.txt`` selects only the GPU runtime: no InsightFace
package, CPU runtime dependency, or post-install cleanup is needed. Choose one
face requirements file per environment, since CPU and GPU runtimes share an import namespace.
When switching from CPU, or if both packages are installed, close pyppbox and
remove both runtimes before reinstalling the GPU requirements::

   python -m pip uninstall -y onnxruntime onnxruntime-gpu
   python -m pip install -r requirements/faces_gpu.txt

The packages share runtime files; reinstalling after both are removed ensures
the GPU package is complete. Include any CUDA-compatible version pin in the
reinstall command, as in the example above, and restart pyppbox afterwards.
No removal is needed in a fresh environment. The GPU runtime also supports
``device=cpu``; installing both runtimes is unnecessary. A provider appearing in
``get_available_providers()`` does not prove that its CUDA libraries can load.
pyppbox also checks the providers of the actual embedding and detector sessions.

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
