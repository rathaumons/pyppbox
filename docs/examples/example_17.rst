Example 17: Compare FaceNet, ArcFace, and AdaFace
=================================================

- **Description:** Compare face extraction coverage, identity classification, cosine scores, and latency using the same identity-folder sample.
- **Featuring:**
   - :py:class:`~pyppbox.modules.reiders.facenet.MyFaceNet`
   - :py:class:`~pyppbox.modules.reiders.onnxface.MyArcFace`
   - :py:class:`~pyppbox.modules.reiders.onnxface.MyAdaFace`
   - Per-backend benchmark reports and saved embeddings

ℹ️ **Source code and input file(s)** -> `{pyppbox repo}/examples`_

.. _{pyppbox repo}/examples: https://github.com/rathaumons/pyppbox/tree/main/examples

This benchmark measures FaceNet, ArcFace, or AdaFace in a separate process.
Install the matching runtime dependencies and obtain the v1.6.0 data wheel for
each model. A dataset directory must contain one image subfolder per identity.
The optional GTA V dataset is installed separately; see :doc:`../getstarted`.

For ArcFace, run from the ``examples/`` directory with explicit installed model paths::

   python example_17_benchmark_faces.py --backend ArcFace --device cpu --dataset /path/to/face_182x182 --detector /path/to/pyppbox/data/modules/arcface/models/det_10g.onnx --model /path/to/pyppbox/data/modules/arcface/models/w600k_r50.onnx --output /path/to/results/arcface

Select ``AdaFace`` and its IR18 ONNX file for AdaFace. FaceNet uses
``--backend FaceNet --device cpu --facenet-data /path/to/pyppbox/data/modules/facenet``
instead of the detector/model arguments. Run models sequentially, using the same
dataset, sample count, thread count, and device where possible.

FaceNet uses CPU in this comparison. ArcFace/AdaFace can use ``--device cuda:0``
with a compatible ONNX Runtime GPU installation. The reports record whichever
CPU/GPU runtime distributions are installed; CPU benchmarking does not require
the GPU distribution.

The default split samples 40 files per identity evenly across sorted paths. The
earlier half fits a new SVM and calibrates a cosine threshold; the later half
tests it. Results include embedding and complete-pipeline latency, detection
failures, known-identity classification, and cosine-pair metrics. They do not
evaluate the full-dataset release classifiers. Files can share capture conditions,
and the test contains no unknown identities, so it does not establish general
recognition accuracy or an unknown-person rejection threshold.

Example source
--------------

.. literalinclude:: ../../examples/example_17_benchmark_faces.py
   :encoding: utf-8
   :language: python
