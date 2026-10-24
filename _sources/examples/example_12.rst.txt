Example 12: Retrain GTA V classifiers for all supported reiders
======================================================================

- **Description:** Use :func:`~pyppbox.standalone.trainReIDClassifier` to retrain GTA V identity classifiers for FaceNet, Torchreid OSNet-AIN, Torchreid OSNet, ArcFace, and AdaFace.
- **Featuring:**
   - :py:func:`~pyppbox.standalone.trainReIDClassifier`

ℹ️ **Source code and input file(s)** -> `{pyppbox repo}/examples`_

.. _{pyppbox repo}/examples: https://github.com/rathaumons/pyppbox/tree/main/examples

Install pyppbox, ``pyppbox-data-gta5``, and the FaceNet, Torchreid, ArcFace, and
AdaFace module data wheels using :doc:`../getstarted`. Running all five
configurations requires Python 3.10+ and one face runtime requirements file:
``requirements/faces.txt`` for CPU or ``requirements/faces_gpu.txt`` for NVIDIA
GPU. Each model needs its own classifier output; they are not interchangeable.

The Torchreid configurations use ``device: cuda``; set both to ``cpu`` if needed.
ArcFace and AdaFace default to ``cpu``. For CUDA, install a compatible ONNX
Runtime GPU build and change their ``device`` values. FaceNet uses TensorFlow's
automatic device selection; ``gpu_mem`` controls its GPU memory allocation.
See :ref:`device-selection` for runtime requirements.

Run from the repository's ``examples/`` directory::

   python example_12_retain_gta5_reid_classifiers.py

The script overwrites five classifier ``.pkl`` files and their companion class-name
``.txt`` files under the installed pyppbox package. Change ``classifier_pkl`` to
a writable output path with an existing parent directory to keep the supplied
classifiers. Remove entries from the training loop to run only the reiders you
have installed. Each backend is closed after training, including on failure.

Torchreid uses the GTA V body images; FaceNet, ArcFace, and AdaFace use the face
images. Only the identity classifiers are trained; neural-network weights remain
frozen. See :doc:`example_18` for training and recognizing a face with an
ArcFace/AdaFace JSON/YAML configuration.

Example source
--------------

.. literalinclude:: ../../examples/example_12_retain_gta5_reid_classifiers.py
   :encoding: utf-8
   :language: python
