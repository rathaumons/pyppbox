17. Compare face recognizers
=============================

This benchmark measures FaceNet, ArcFace, or AdaFace in a separate process.
Install the matching runtime dependencies and obtain the v1.6.0 data wheel for
each model. A dataset directory must contain one image subfolder per identity.
The optional GTA V dataset is installed separately; see :doc:`../getstarted`.

For ArcFace, run from the repository root with explicit installed model paths::

   python examples/example_17_benchmark_faces.py --backend ArcFace --device cpu --dataset /path/to/face_182x182 --detector /path/to/pyppbox/data/modules/arcface/models/det_10g.onnx --model /path/to/pyppbox/data/modules/arcface/models/w600k_r50.onnx --output /path/to/results/arcface

Select ``AdaFace`` and its IR18 ONNX file for AdaFace. FaceNet uses
``--backend FaceNet --device cpu --facenet-data /path/to/pyppbox/data/modules/facenet``
instead of the detector/model arguments. Run models sequentially, using the same
dataset, sample count, thread count, and device where possible.

The default split samples 40 files per identity evenly across sorted paths. The
earlier half fits a new SVM and calibrates a cosine threshold; the later half
tests it. Results include embedding and complete-pipeline latency, detection
failures, known-identity classification, and cosine-pair metrics. They do not
evaluate the full-dataset release classifiers. Files can share capture conditions,
and the test contains no unknown identities, so it does not establish general
recognition accuracy or an unknown-person rejection threshold.

.. literalinclude:: ../../examples/example_17_benchmark_faces.py
   :language: python
