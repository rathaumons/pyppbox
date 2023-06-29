.. _supportedmodules-page:

Supported Modules
=================

The table below shows all the current supported modules which are integrated in **pyppbox** in many different ways. The main idea is to make them fully compatible with the needs of **PoseTReID framework** which focuses more on the real-life applications. Thus, **not every original feature of every supported module is fully functioning as it was made**. For example, the supported `YOLO_Classic` / `YOLO` module in the table can only be used as a detector (Inference/Prediction) using the native or official **`.weights`** files, but it can't do any custom data traning, etc.

| 

.. table:: 
   :widths: auto

   +------------+--------------+------------------+-------------------------------------------------+
   | Modules    | General Name | Config Name      | Details                                         |
   +============+==============+==================+=================================================+
   | Detectors  | YOLO         | YOLO_Classic     | | * Built-in by using OpenCV DNN                |
   |            |              |                  | | * Model: .weights `V2, V3`_, V4_              |
   |            |              |                  | | * Run on: CPU or GPU (OpenCV DNN)             |
   |            |              +------------------+-------------------------------------------------+
   |            |              | YOLO_Ultralytics | | * Integrated by linking `ultralytics`         |
   |            |              |                  | | * Model: .pt `V3, V5, V8`_                    |
   |            |              |                  | | * Run on: CPU or GPU (PyTorch)                |
   +------------+--------------+------------------+-------------------------------------------------+
   | Trackers   | Centroid     | Centroid         | | * Built-in / Native                           |
   |            |              |                  | | * Run on: CPU                                 |
   |            +--------------+------------------+-------------------------------------------------+
   |            | SORT         | SORT             | | * Integrated by embedding                     |
   |            |              |                  | | * `SORT repo`_                                |
   |            |              |                  | | * Run on: CPU                                 |
   |            +--------------+------------------+-------------------------------------------------+
   |            | DeepSORT     | DeepSORT         | | * Integrated by embedding                     |
   |            |              |                  | | * `DeepSORT repo`_                            |
   |            |              |                  | | * Run on: CPU or GPU (Tensorflow)             |
   +------------+--------------+------------------+-------------------------------------------------+
   | ReIDers    | FaceNet      | FaceNet          | | * Integrated by embedding                     |
   |            |              |                  | | * `FaceNet repo`_                             |
   |            |              |                  | | * Run on: CPU or GPU (Tensorflow)             |
   |            +--------------+------------------+-------------------------------------------------+
   |            | Torchreid    | Trochreid        | | * Integrated by linking `pyppbox-torchreid`_  |
   |            |              |                  | | * Model: OSNet-AIN, OSNet, MLFN               |
   |            |              |                  | | * Run on: GPU-Only (PyTorch)                  |
   +------------+--------------+------------------+-------------------------------------------------+

.. _V2, V3: https://pjreddie.com/darknet/yolo/
.. _V4: https://github.com/AlexeyAB/darknet
.. _V3, V5, V8: https://github.com/ultralytics/assets/releases
.. _SORT repo: https://github.com/abewley/sort
.. _DeepSORT repo: https://github.com/deshwalmahesh/yolov7-deepsort-tracking
.. _FaceNet repo: https://github.com/davidsandberg/facenet
.. _pyppbox-torchreid: https://github.com/rathaumons/torchreid-for-pyppbox

| 

.. toctree::
   :maxdepth: 2

   modules/detectors
   modules/trackers
   modules/reiders

|