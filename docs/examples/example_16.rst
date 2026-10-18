Example 16: Export an AdaFace embedding model to ONNX
=====================================================

- **Description:** Reproduce the packaged AdaFace IR18 VGG2 export and compare its ONNX embeddings with the original PyTorch model.
- **Featuring:**
   - CVLFace/AdaFace checkpoint loading
   - ``torch.onnx.export``
   - ONNX Runtime inference and numerical comparison
   - Model hashes and preprocessing metadata

ℹ️ **Source code and input file(s)** -> `{pyppbox repo}/examples`_

.. _{pyppbox repo}/examples: https://github.com/rathaumons/pyppbox/tree/dev/examples

The v1.6.0 ``pyppbox-data-adaface`` wheel already contains a verified IR18 VGG2
ONNX export. This advanced example reproduces the export using the author's
CVLFace checkpoint and the original AdaFace architecture. It does not download
weights or train a network.

Install the ``faces`` extra and ``onnx``, obtain the
`CVLFace IR18 VGG2 checkpoint <https://huggingface.co/minchul/cvlface_adaface_ir18_vgg2>`_,
and clone the `original AdaFace architecture <https://github.com/mk-minchul/AdaFace>`_.
Run from the ``examples/`` directory::

   python example_16_export_adaface_onnx.py --adaface-repo /path/to/AdaFace --checkpoint /path/to/model.pt --output /path/to/adaface_ir18_vgg2.onnx

The export uses RGB NCHW input normalized with ``(pixel - 127.5) / 127.5`` and
compares ONNX outputs against PyTorch. The JSON sidecar records checkpoint/export
hashes and numerical discrepancy. The model and dataset terms still apply.
See :doc:`../pyppbox/modules/reiders` before changing model preprocessing or
reusing an identity classifier with a different export.

Example source
--------------

.. literalinclude:: ../../examples/example_16_export_adaface_onnx.py
   :encoding: utf-8
   :language: python
