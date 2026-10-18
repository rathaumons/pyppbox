Example 15: Use RF-DETR detection and SORT tracking
===================================================

- **Description:** Detect people with RF-DETR and track them with SORT using the same pipeline and Person objects as the earlier examples.
- **Featuring:**
   - :py:meth:`~pyppbox.ppb.mt.MT.setMainDetector`
   - :py:meth:`~pyppbox.ppb.mt.MT.setMainTracker`
   - :py:meth:`~pyppbox.ppb.mt.MT.detectPeople`
   - :py:meth:`~pyppbox.ppb.mt.MT.trackPeople`
   - :py:func:`~pyppbox.utils.visualizetools.visualizePeople`

ℹ️ **Source code and input file(s)** -> `{pyppbox repo}/examples`_

.. _{pyppbox repo}/examples: https://github.com/rathaumons/pyppbox/tree/dev/examples

``RF_DETR`` is an optional detector included in pyppbox V4, using the existing
configuration files, public pipeline, and GUI. Python **3.10+** is required for
RF-DETR; existing detectors remain usable without installing this backend.

After installing pyppbox, install the optional requirements from the repository root::

   python -m pip install -r requirements/rfdetr.txt

Then change to the ``examples/`` directory and run with its bundled video::

   cd examples
   python example_15_rfdetr.py --device cpu --frames 30

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

The GUI's device dropdown is editable: select ``0``, ``cpu``, ``cuda:0``, ``mps``,
or ``auto``, or type another valid GPU index / ``cuda:N`` value. ``auto`` delegates
selection to RF-DETR. Other modules use their own presets and automatic-selection
rules; see :ref:`device-selection`.

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

Downloading pretrained weights
------------------------------

Choose one of the four detection checkpoints supported by this adapter. These
direct links are from RF-DETR 1.10's published model registry:

.. list-table:: Pretrained RF-DETR detection checkpoints
   :header-rows: 1
   :widths: 12 12 35 41

   * - ``model_size``
     - Resolution
     - Save as
     - Download
   * - ``nano``
     - 384
     - ``rf-detr-nano.pth``
     - `Nano checkpoint <https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth>`_
   * - ``small``
     - 512
     - ``rf-detr-small.pth``
     - `Small checkpoint <https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth>`_
   * - ``medium``
     - 576
     - ``rf-detr-medium.pth``
     - `Medium checkpoint <https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth>`_
   * - ``large``
     - 704
     - ``rf-detr-large-2026.pth``
     - `Large checkpoint <https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth>`_

The first three URLs all end in ``checkpoint_best_regular.pth``; rename each
download to the corresponding filename above. For example, download Nano into
your working directory with::

   curl -L --fail -o rf-detr-nano.pth https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth

Browse to the file in the RF-DETR settings panel and select its matching size.
Alternatively, the optional `pyppbox-data-rfdetr v1.6.0 wheel
<https://github.com/rathaumons/pyppbox-data/releases/download/v1.6.0/pyppbox_data_rfdetr-1.6.0-py3-none-any.whl>`_ installs all four
checkpoints under ``{pyppbox root}/data/modules/rf_detr/``, matching the existing
internal config paths. See :doc:`../getstarted` for installation commands.
Installing a data wheel does not
install the RF-DETR runtime; install ``requirements/rfdetr.txt`` as shown above too.

For data-package details, see
`pyppbox-data <https://github.com/rathaumons/pyppbox-data/>`_.

For offline use, install the runtime dependencies and data wheel in advance, or
select an existing local checkpoint. These four complete checkpoints were tested
with Python network access blocked. Custom model architectures may have additional
assets to prepare separately. Never select a generated CI fixture as a production
model.

Roboflow's `pretrained aliases`_ include more models for its Inference service.
Those service model IDs are not checkpoint paths for this adapter; xlarge,
2xlarge, segmentation, and keypoint-preview models are not exposed here.

Automatic downloading and cache
-------------------------------

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
.. _pretrained aliases: https://docs.roboflow.com/models/pretrained-aliases#rf-detr

Example source
--------------

.. literalinclude:: ../../examples/example_15_rfdetr.py
   :encoding: utf-8
   :language: python

Validation
----------

Run the following validation commands from the repository root.

The core suite exercises the adapter, configuration fallback, and pipeline with
deterministic model boundaries, without downloading RF-DETR::

   python -m unittest discover -s tests -v

For the offscreen RF-DETR settings dialog, with PyQt6 installed::

   python -m unittest discover -s tests/integration -p test_rfdetr_gui.py -v

For real CPU/CUDA inference, install the RF-DETR requirements above and set
``PYPPBOX_RUN_RFDETR=1`` before running::

   python -m unittest discover -s tests/integration -p test_rfdetr_model.py -v

The opt-in test downloads/caches Nano weights if absent and uses the bundled GTA
video. CUDA is tested when available. Published upstream TensorRT benchmarks do
not establish the performance of this PyTorch adapter; compare complete pipelines
on representative clips before selecting a detector.

The dedicated CPU workflow validates all four variants on Linux x86_64, Windows
AMD64, and macOS ARM with Python 3.11, RF-DETR 1.10.0, and Torch 2.5.1. It compares
native/prepared predictions and frozen/current pyppbox wheels, including class
filters, SORT gaps/recovery, rendered output, offline loading, missing/corrupt
checkpoints, unavailable devices, reset/close, and independent streams. Generated
two-class native and Lightning checkpoints exercise class-head loading and numeric
class IDs; they do not establish trained-model accuracy or support every custom
architecture. GUI process cleanup is tested with actual owned subprocess trees on
all three systems. Earlier Nano CUDA checks remain separate from CPU qualification.
