Example 15: Use RF-DETR detection and SORT tracking
===================================================

Install pyppbox, ``requirements/rfdetr.txt``, and the optional
``pyppbox-data-rfdetr`` wheel as described in :doc:`../getstarted`.
From the repository's ``examples/`` directory::

    python example_15_rfdetr.py --device cpu --frames 30

Use ``--device cuda:0`` for NVIDIA GPU inference, ``--show`` for a video window,
or ``--video path/to/video.mp4`` for another clip. ReID is disabled in this example.

Configuration
-------------

``RF_DETR`` supports Nano, Small, Medium, and Large detection checkpoints.
The default is Nano on GPU 0, using a local model file::

    dt_name: RF_DETR
    model_size: nano
    model_file: data/modules/rf_detr/rf-detr-nano.pth
    device: 0
    conf: 0.4
    max_det: 100
    person_class_id: 1
    optimize: True
    repspoint_calibration: 0.25

Select a checkpoint matching ``model_size``. Set ``person_class_id`` to the
native person class ID for a custom model; the supplied COCO checkpoints use 1.
``optimize`` prepares the model for inference. See :doc:`../pyppbox/config`
for path and device rules.

Downloading pretrained weights
------------------------------

The data wheel supplies these files under ``data/modules/rf_detr/`` in the
installed package. For manual setup, download a checkpoint and select it in
the RF-DETR settings panel.

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

Nano, Small, and Medium share the server filename ``checkpoint_best_regular.pth``;
rename each download as shown above. Set ``model_file: ''`` to let RF-DETR
download and cache weights automatically (``RF_HOME`` controls the cache).
For offline use, install the data wheel or select a local checkpoint in advance.

Example source
--------------

.. literalinclude:: ../../examples/example_15_rfdetr.py
   :encoding: utf-8
   :language: python
