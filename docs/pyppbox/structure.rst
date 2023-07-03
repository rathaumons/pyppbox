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
   ├───modules  ..........................  All supported detector/trackers/reiders
   │   │   __init__.py
   │   ├───detectors  ....................  All supported detectors
   │   │   │   __init__.py
   │   │   ├───yolocls  ..................  Detector YOLO_Classic
   │   │   │       __init__.py
   │   │   └───yoloult  ..................  Detector YOLO_Ultralytics
   │   │           __init__.py
   │   ├───trackers  .....................  All supported trackers
   │   │   │   __init__.py
   │   │   ├───centroid  .................  Tracker Centroid
   │   │   │       __init__.py
   │   │   ├───sort  .....................  Tracker SORT
   │   │   │       __init__.py
   │   │   └───deepsort  .................  Tracker DeepSORT
   │   │           __init__.py
   │   └───reiders  ......................  All supported reiders
   │       │   __init__.py
   │       ├───facenet  ..................  Reider FaceNet
   │       │       __init__.py
   │       └───torchreid  ................  Reider Torchreid
   │               __init__.py
   ├───standalone  .......................  Standalone functions
   │       __init__.py
   └───utils  ............................  Utilities including visualization, evaulation, etc.
           __init__.py


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
       ├───logs  .........................  Log directory, part of pyppbox-data module
       ├───modules  ......................  Data directory, part of pyppbox-data module
       │   ├───deepsort  .................  Directory, consists of all data for DeepSORT
       │   │       mars-small128.pb
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
