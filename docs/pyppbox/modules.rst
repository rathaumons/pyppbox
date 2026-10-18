.. _supportedmodules-page:

Supported Modules
=================

The tables below list the detectors, trackers, and ReIDers integrated in
**pyppbox** for the **PoseTReID framework** (`read more here`_). Each integration
exposes the features needed by the pipeline, rather than every feature of the
original project. For example, ``YOLO_Classic`` uses official ``.weights`` files
for detection; it does not train YOLO models.

Config names are the values used in ``dt_name``, ``tk_name``, or ``ri_name``.
CPU/GPU entries describe backend capabilities. Device controls and automatic
selection differ by module; see :ref:`device-selection` for editable dropdowns,
presets, runtime requirements, and the OpenCV/TensorFlow automatic-device rows.

.. list-table:: Supported Detectors
   :header-rows: 1
   :widths: 16 20 64

   * - General Name
     - Config Name
     - Details
   * - YOLO Classic
     - ``YOLO_Classic``
     - * Integration: built-in OpenCV DNN detector
       * Models: ``.weights`` files for `V2, V3`_ and `V4`_
       * Runtime: CPU or GPU; OpenCV selects the device automatically
   * - YOLO Ultralytics
     - ``YOLO_Ultralytics``
     - * Integration: linked `ultralytics`_ package
       * Models: ``.pt`` files for `V3, V5, V8, V9, v10, v11, v12, v26`_
       * Runtime: PyTorch; configurable device
   * - RF-DETR
     - ``RF_DETR``
     - * Integration: optional ``requirements/rfdetr.txt`` on Python 3.10+
       * Models: RF-DETR Nano, Small, Medium, and Large
       * Runtime: PyTorch; configurable device
       * Example: :doc:`../examples/example_15`
   * - Ground truth
     - ``GT``
     - * Integration: built-in ground-truth reader
       * Input: detections and identities from a ground-truth text file
       * Runtime: CPU
       * Example: :doc:`../examples/example_10`

For ``GT``, set both tracker and reider to ``"None"`` to preserve the supplied
identities automatically. With other stages enabled,
:meth:`pyppbox.ppb.mt.MT.forceFullGTMode` overrides detection-only GT behavior until
the next module selection.

.. list-table:: Supported Trackers
   :header-rows: 1
   :widths: 16 20 64

   * - General Name
     - Config Name
     - Details
   * - Centroid
     - ``Centroid``
     - * Integration: built-in centroid tracker
       * Runtime: CPU
   * - SORT
     - ``SORT``
     - * Integration: embedded `SORT repo`_
       * Runtime: CPU
   * - DeepSORT
     - ``DeepSORT``
     - * Integration: embedded `DeepSORT repo`_
       * Runtime: TensorFlow appearance encoder; CPU assignment
       * Device: TensorFlow selects devices automatically

.. list-table:: Supported ReIDers
   :header-rows: 1
   :widths: 16 20 64

   * - General Name
     - Config Name
     - Details
   * - FaceNet
     - ``FaceNet``
     - * Integration: embedded `FaceNet repo`_
       * Models: MTCNN face detection, FaceNet embeddings, and an identity SVM
       * Runtime: CPU or GPU; TensorFlow selects devices automatically
       * Classifier training: :doc:`../examples/example_12`
   * - Torchreid
     - ``Torchreid``
     - * Integration: linked `pyppbox-torchreid`_ package
       * Models: OSNet-AIN, OSNet, and MLFN embeddings with an identity SVM
       * Runtime: PyTorch; configurable device
       * Classifier training: :doc:`../examples/example_12`
   * - ArcFace
     - ``ArcFace``
     - * Integration: optional ``requirements/faces.txt`` or ``requirements/faces_gpu.txt`` on Python 3.10+
       * Models: SCRFD alignment and local ArcFace R50 ONNX embeddings
       * Runtime: ONNX Runtime on CPU or NVIDIA CUDA; configurable device
       * Identity SVM: separate ArcFace classifier trained with frozen embeddings
       * Examples: :doc:`../examples/example_17`, :doc:`../examples/example_18`
   * - AdaFace
     - ``AdaFace``
     - * Integration: optional ``requirements/faces.txt`` or ``requirements/faces_gpu.txt`` on Python 3.10+
       * Models: SCRFD alignment and local AdaFace IR18 ONNX embeddings
       * Runtime: ONNX Runtime on CPU or NVIDIA CUDA; configurable device
       * Identity SVM: separate AdaFace classifier trained with frozen embeddings
       * Examples: :doc:`../examples/example_16`, :doc:`../examples/example_17`, :doc:`../examples/example_18`

ArcFace and AdaFace use their respective v1.6.0 data wheels for the supplied
models and classifiers. See :doc:`modules/reiders` for installation, custom
models, classifier compatibility, and training requirements.

.. _read more here: https://github.com/rathaumons/PoseTReID_DATASET#-posetreid
.. _V2, V3: https://pjreddie.com/darknet/yolo/
.. _V4: https://github.com/AlexeyAB/darknet
.. _ultralytics: https://github.com/rathaumons/ultralytics-for-pyppbox
.. _V3, V5, V8, V9, v10, v11, v12, v26: https://github.com/ultralytics/assets/releases
.. _SORT repo: https://github.com/abewley/sort
.. _DeepSORT repo: https://github.com/deshwalmahesh/yolov7-deepsort-tracking
.. _FaceNet repo: https://github.com/davidsandberg/facenet
.. _pyppbox-torchreid: https://github.com/rathaumons/torchreid-for-pyppbox

.. toctree::
   :maxdepth: 2

   modules/detectors
   modules/trackers
   modules/reiders
