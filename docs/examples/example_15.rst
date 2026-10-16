15. RF-DETR detection and tracking
==================================

``RF_DETR`` is an optional detector for the V3.16 transition release, using the existing
configuration files, public pipeline, and GUI. Python **3.10+** is required for
RF-DETR; existing detectors remain usable without installing this extra.

Install into a virtual environment from this checkout::

   python -m pip install ".[rfdetr]"

For a release package that includes RF-DETR, use ``python -m pip install
"pyppbox[rfdetr]"``. Earlier published versions do not provide this extra.

Run from the repository root, using its bundled video::

   python -m examples.example_15_rfdetr --device cpu --frames 30

Use ``--device cuda:0`` for a compatible CUDA PyTorch installation. Add ``--show``
for an annotated video window, or ``--video path/to/video.mp4`` for another clip.
The example uses SORT tracking and disables ReID, so no ReID model files are needed.

Model and config choices
------------------------

The adapter targets ``rfdetr>=1.10.0,<1.11.0`` and provides the Apache-designated
**Nano, Small, Medium, and Large detection models** from `RF-DETR`_. Each model
uses its published native resolution. Segmentation, keypoint-preview, Plus models,
training, ONNX, and TensorRT export are outside this initial integration.

The existing ``detectors.yaml`` format accepts this additional document:

.. code-block:: yaml

   dt_name: RF_DETR
   conf: 0.4
   model_size: nano
   device: 0
   max_det: 100
   model_file: data/modules/rf_detr/rf-detr-nano.pth
   person_class_id: 1
   optimize: True
   repspoint_calibration: 0.25

Like the YOLO defaults, the bundled config selects GPU ``device: 0`` and a local
checkpoint under ``data/modules``. Place the Nano weights at
``{pyppbox root}/data/modules/rf_detr/rf-detr-nano.pth`` for internal configs, or
select your downloaded checkpoint using the GUI. Set ``device: cpu`` when CUDA
is unavailable. Integer GPU indices are saved as integers and passed to RF-DETR
as ``cuda:N``.

A nonempty ``model_file`` must be an existing local checkpoint matching
``model_size``. Paths follow the existing pyppbox rules: package-relative for
internal configs, working-directory-relative for custom configs or mappings.
When a mapping omits ``model_file``, its local default filename follows
``model_size``. The GUI also updates a standard default path when switching sizes;
a custom checkpoint path is retained. Model loading is deferred until selection.

An explicitly empty ``model_file: ''`` lets RF-DETR download its pretrained
checkpoint on first use instead. Set ``RF_HOME`` to choose its cache directory.

With the supported RF-DETR 1.10 release, the default cache is
``~/.roboflow/models`` (on Windows, under your user profile). Nano, Small, Medium,
and Large use ``rf-detr-nano.pth``, ``rf-detr-small.pth``, ``rf-detr-medium.pth``,
and ``rf-detr-large-2026.pth``, respectively. Keep ``model_file`` empty when
switching model sizes to select the matching pretrained checkpoint automatically.
For a local checkpoint, use the GUI's browse button and select its matching
``model_size``.

``person_class_id`` is **1** for RF-DETR's pretrained COCO models. Fine-tuned models
can use a different ID (often **0**); set it explicitly for your checkpoint.
General ``MyRFDETR.detect()`` class filters also use native RF-DETR IDs.

The wrapper accepts BGR uint8 frames, converts them to RGB for RF-DETR, and returns
the same BGR array. Output is the existing ``Person`` box representation, with
confidence in [0, 1] and empty keypoints. Boxes are clipped to the image; class and
width filters run before the confidence-ordered ``max_det`` limit. No extra NMS or
IoU parameter is added to RF-DETR's native predictions.

``optimize=True`` calls RF-DETR's inference preparation once, in place, with
compilation disabled. This avoids keeping a second model copy. The wrapped model
is then for inference only; use a separate RF-DETR instance for training/export.

Old configuration directories require no migration: RF-DETR defaults are available
in memory when its document is absent. Selecting it by name or opening the GUI
uses those defaults; saving the RF-DETR dialog adds its document. Existing detector
dialogs preserve the other detector documents when saving.

.. _RF-DETR: https://github.com/roboflow/rf-detr

Example source
--------------

.. literalinclude:: ../../examples/example_15_rfdetr.py
   :language: python

Validation
----------

The core suite exercises the adapter, configuration fallback, and pipeline with
deterministic model boundaries, without downloading RF-DETR::

   python -m unittest discover -s tests -v

For the offscreen RF-DETR settings dialog, with PyQt6 installed::

   python -m unittest discover -s tests/integration -p test_rfdetr_gui.py -v

For real CPU/CUDA inference, install the RF-DETR extra and set
``PYPPBOX_RUN_RFDETR=1`` before running::

   python -m unittest discover -s tests/integration -p test_rfdetr_model.py -v

The opt-in test downloads/caches Nano weights if absent and uses the bundled GTA
video. CUDA is tested when available. Published upstream TensorRT benchmarks do
not establish the performance of this PyTorch adapter; compare complete pipelines
on representative clips before selecting a detector. Local real-model validation
has covered Nano on CPU and CUDA; the other model sizes, custom checkpoints, and
other devices require their own runtime validation.
