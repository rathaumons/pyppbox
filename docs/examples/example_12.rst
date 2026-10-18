Example 12: Retrain FaceNet and both Torchreid classifiers for GTA V
====================================================================

- **Description:** Use :func:`~pyppbox.standalone.trainReIDClassifier` to retrain the FaceNet, Torchreid OSNet-AIN, and Torchreid OSNet GTA V identity classifiers.
- **Featuring:**
   - :py:func:`~pyppbox.standalone.trainReIDClassifier`

ℹ️ **Source code and input file(s)** -> `{pyppbox repo}/examples`_

.. _{pyppbox repo}/examples: https://github.com/rathaumons/pyppbox/tree/dev/examples

The script overwrites these three configured classifiers using the supplied
training images and frozen embedding models. It does not retrain ArcFace or
AdaFace; see :doc:`example_18` for those model-specific identity classifiers.

Example source
--------------

.. literalinclude:: ../../examples/example_12_retain_gta5_reid_classifiers.py
   :encoding: utf-8
   :language: python
