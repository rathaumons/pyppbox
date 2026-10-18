18. Recognize faces and train identity classifiers
===================================================

Install ``.[faces]`` from this checkout and either the ArcFace or AdaFace v1.6.0
data wheel. The supplied GTA V classifier training environment is documented in
:doc:`../pyppbox/modules/reiders`.
For direct inference with the packaged defaults, see
:doc:`../pyppbox/modules/reiders`. This command-line example accepts a JSON/YAML
configuration for custom model paths and identity classifiers.

Save an ``arcface.json`` file with absolute paths matching your installation::

   {
     "ri_name": "ArcFace",
     "model_det": "/path/to/pyppbox/data/modules/arcface/models/det_10g.onnx",
     "model_file": "/path/to/pyppbox/data/modules/arcface/models/w600k_r50.onnx",
     "classifier_pkl": "/path/to/pyppbox/data/modules/arcface/classifier/gta5.pkl",
     "device": "cpu"
   }

Run from the repository root::

   python examples/example_18_onnx_face_reid.py --config arcface.json --image /path/to/face.jpg

For AdaFace, use ``"ri_name": "AdaFace"`` and the detector, IR18 ONNX model, and
classifier under ``data/modules/adaface``. The command prints the identity and
SVM confidence percentage. A probability threshold does not provide a calibrated
unknown-person probability.

To train your own identities, set ``train_data`` to a directory containing one
image subfolder per person and ``classifier_pkl`` to a new writable output file.
Create its parent directory first. Add ``--train`` to the command to train, reload,
and run inference. Training fits an identity SVM using frozen embeddings, skips
images with no face, and rejects an identity without usable faces. Model and
preprocessing fingerprints prevent incompatible classifier reuse.

.. literalinclude:: ../../examples/example_18_onnx_face_reid.py
   :language: python
