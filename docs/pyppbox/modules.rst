.. _supportedmodules-page:

Supported Modules
=================

The table below shows all the current supported modules which are integrated in **pyppbox** in many different ways. The main idea of the integrations is to make them fully compatible with our **PoseTReID framework** `read more here`_. Thus, **not every original feature of every supported module is fully functioning as it was made**. For example, the supported `YOLO_Classic` / `YOLO` module in the table can only be used as a detector (Inference/Prediction) using the native or official **`.weights`** files, but it can't do any custom data traning, etc.

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
   |            |              | YOLO_Ultralytics | | * Integrated by linking `ultralytics`_        |
   |            |              |                  | | * Model: .pt `V3, V5, V8, V9`_                |
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
   |            | Torchreid    | Torchreid        | | * Integrated by linking `pyppbox-torchreid`_  |
   |            |              |                  | | * Model: OSNet-AIN, OSNet, MLFN               |
   |            |              |                  | | * Run on: CPU or GPU (PyTorch)                |
   +------------+--------------+------------------+-------------------------------------------------+

.. _read more here: https://github.com/rathaumons/PoseTReID_DATASET#-posetreid
.. _V2, V3: https://pjreddie.com/darknet/yolo/
.. _V4: https://github.com/AlexeyAB/darknet
.. _ultralytics: https://github.com/rathaumons/ultralytics-for-pyppbox
.. _V3, V5, V8, V9: https://github.com/ultralytics/assets/releases
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
