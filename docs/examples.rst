💡 Examples
===========

The examples below demonstrate **pyppbox** configuration, detection, tracking,
ReID, evaluation, and optional modules. Install the package and required model/data
wheels first; see :ref:`getstarted-page`.

Examples 1–14 use the standalone/direct-module interfaces; example 12 also
includes the V4 ArcFace/AdaFace reiders. Examples 15–18 cover additional V4
workflows. Install each example's optional requirements from the repository
root before changing directories.

All example scripts use the repository's ``examples/`` directory as their working
directory. From the repository root::

   cd examples
   python example_01_basic.py

For the newer command-line examples, use the same working directory::

   python example_15_rfdetr.py --device cpu --frames 30
   python example_16_export_adaface_onnx.py --help
   python example_17_benchmark_faces.py --help
   python example_18_onnx_face_reid.py --help

Relative ``data/``, ``cfg/``, ``single_config/``, configuration-file, and output
paths resolve from this working directory. Examples 3, 4, 5, 11, and 13 contain
``C:/pyppbox_v3/`` placeholders in their scripts or custom configs; replace these
with your installed ``pyppbox`` directory. Find it with::

   python -c "import os, pyppbox; print(os.path.dirname(pyppbox.__file__))"

Example 12 retrains the FaceNet, Torchreid OSNet-AIN, Torchreid OSNet, ArcFace,
and AdaFace GTA V classifiers. Example 18 trains and recognizes a face using a
custom ArcFace/AdaFace configuration. Model requirements are documented in
:doc:`pyppbox/modules/reiders`. Device settings
and automatic selection are described in :ref:`device-selection`.

ℹ️ Source code and **required input files** are available here:
`{pyppbox repo}/examples`_.

.. _{pyppbox repo}/examples: https://github.com/rathaumons/pyppbox/tree/main/examples

.. toctree::
   :maxdepth: 1

   examples/example_01
   examples/example_02
   examples/example_03
   examples/example_04
   examples/example_05
   examples/example_06
   examples/example_07
   examples/example_08
   examples/example_09
   examples/example_10
   examples/example_11
   examples/example_12
   examples/example_13
   examples/example_14
   examples/example_15
   examples/example_16
   examples/example_17
   examples/example_18

|

