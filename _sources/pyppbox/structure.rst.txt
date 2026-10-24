.. _structure-page:

Structure
=========

Base Structure of ``pyppbox``
-----------------------------

.. code-block:: text

   pyppbox  ..............................  Root
   ├───config
   │   ├───cfg  ..........................  Internal config directory
   │   │       main.yaml  ................  Internal config file for setting main modules
   │   │       detectors.yaml  ...........  Internal config file for all supported detectors
   │   │       trackers.yaml  ............  Internal config file for all supported trackers
   │   │       reiders.yaml  .............  Internal config file for all supported reiders
   │   └───strings
   │           strings.yaml  .............  Internal config file for unified strings
   ├───data  .............................  Internal data directory, pyppbox-data module
   ├───gui
   │   ├───configwidgets.py  .............  Shared forms, path rows, and device dropdowns
   │   ├───linklabel.py  .................  Theme-aware launcher link
   │   ├───ui_launcher.py  ...............  Module selection and demo launcher
   │   └───ui_*.py  ......................  Module configuration dialogs
   ├───modules  ..........................  All supported detector/trackers/reiders
   │   ├───detectors  ....................  All supported detectors
   │   │   ├───yolocls  ..................  Detector YOLO_Classic
   │   │   ├───yoloult  ..................  Detector YOLO_Ultralytics
   │   │   └───rfdetr  ...................  Optional detector RF_DETR
   │   ├───trackers  .....................  All supported trackers
   │   │   ├───centroid  .................  Tracker Centroid
   │   │   ├───sort  .....................  Tracker SORT
   │   │   └───deepsort  .................  Tracker DeepSORT
   │   └───reiders  ......................  All supported reiders
   │       ├───facenet  ..................  Reider FaceNet
   │       ├───torchreid  ................  Reider Torchreid
   │       └───onnxface  .................  Optional ArcFace and AdaFace adapters
   ├───ppb  ..............................  Pipeline API and per-instance MT class
   ├───standalone  .......................  Public facade for the shared pipeline
   └───utils  ............................  Utilities including visualization, evaluation, etc.


Data: ``pyppbox-data`` and ``pyppbox-data-gta5``
------------------------------------------------

.. code-block:: text

   pyppbox
   └───data  .............................  Internal data directory, pyppbox-data module
       ├───datasets
       │   └───GTA_V_DATASET  ............  PoseTReID's GTA_V_DATASET, pyppbox-data-gta5
       │       ├───body_128x256  .........  Cropped & classified body boxes in 128x256
       │       │   ├───Amanda
       │       │   │       *.jpg
       │       │   ├───Franklin
       │       │   │       *.jpg
       │       │   ├───Lester
       │       │   │       *.jpg
       │       │   ├───Michael
       │       │   │       *.jpg
       │       │   └───Trevor
       │       │           *.jpg
       │       ├───face_182x182  .........  Cropped & classified face boxes in 182x182
       │       │   ├───Amanda
       │       │   │       *.png
       │       │   ├───Franklin
       │       │   │       *.png
       │       │   ├───Lester
       │       │   │       *.png
       │       │   ├───Michael
       │       │   │       *.png
       │       │   └───Trevor
       │       │           *.png
       │       ├───ground_truth  .........  Ground-truth directory
       │       │       gt_map.txt  .......  VIDEO:TXT mapping file
       │       │       *.txt  ............  Ground-truth  .txt file
       │       │       *.csv  ............  Ground-truth  .csv file
       │       └───videos
       │               *.mp4
       ├───logs  .........................  Legacy logs, retained for compatibility
       ├───modules  ......................  Data directory, part of pyppbox-data module
       │   ├───arcface  ..................  Optional pyppbox-data-arcface v1.6.0
       │   │   ├───classifier
       │   │   │       gta5.pkl
       │   │   └───models
       │   │           det_10g.onnx
       │   │           w600k_r50.onnx
       │   ├───adaface  ..................  Optional pyppbox-data-adaface v1.6.0
       │   │   ├───classifier
       │   │   │       gta5.pkl
       │   │   └───models
       │   │           det_10g.onnx
       │   │           adaface_ir18_vgg2.onnx
       │   ├───deepsort  .................  Directory, consists of all data for DeepSORT
       │   │       mars-small128.pb
       │   ├───rf_detr  ..................  Optional pyppbox-data-rfdetr v1.6.0
       │   │       rf-detr-nano.pth
       │   │       rf-detr-small.pth
       │   │       rf-detr-medium.pth
       │   │       rf-detr-large-2026.pth
       │   ├───facenet  ..................  Directory, consists of all data for FaceNet
       │   │   ├───classifier
       │   │   │       *.pkl
       │   │   │       *.txt
       │   │   └───models
       │   │       ├───20180402-114759
       │   │       │       20180402-114759.pb
       │   │       │
       │   │       └───det
       │   │               *.npy
       │   ├───torchreid  ................  Directory, consists of all data for Torchreid
       │   │   ├───classifier
       │   │   │       *.pkl
       │   │   │       *.txt
       │   │   └───models
       │   │       ├───base
       │   │       │       *.pth
       │   │       └───torchreid
       │   │               *.tar
       │   ├───yolo_classic  .............  Directory, consists of all data for YOLO_Classic
       │   │       coco.names
       │   │       *.cfg
       │   │       *.weights
       │   └───yolo_ultralytics  .........  Directory, consists of all data for YOLO_Ultralytics
       │           *.pt
       └───res

Mutable runtime files
---------------------

New logs and GUI state are written outside the installed package, under the
per-user directory described in :doc:`utils`. Set ``PYPPBOX_RUNTIME_DIR`` to choose
another writable location. GUI launch handoffs are isolated per launch; installed
``gui/tmp`` files serve only as legacy defaults. Configuration, asset, and explicit
result-export paths retain their existing behavior.

All eight module data wheels use version **1.6.0** for V4; the independently
versioned GTA V dataset is unchanged. See :doc:`../getstarted` for installation and
:doc:`../examples/example_15` for direct RF-DETR weight downloads. Data wheels
supply assets only and do not install their inference backends.
See :doc:`modules/reiders` for the model-specific ONNX paths.
