.. _supportedmodules-page:

Supported Modules
=================

The tables below show all the currently supported modules integrated in **pyppbox** in different ways. The main goal is to make them fully compatible with our **PoseTReID framework** `read more here`_. Thus, **not every original feature of each supported module is fully functioning as originally made**.  
For example, the supported `YOLO_Classic` / `YOLO` module can only be used as a detector (Inference/Prediction) with official **`.weights`** files, but not for custom data training, etc.

|

.. table:: Supported Detectors

   +--------------+------------------+----------------------------------------------------------+
   | General Name | Config Name      | Details                                                  |
   +==============+==================+==========================================================+
   | YOLO         | YOLO_Classic     | * Built-in by using OpenCV DNN                           |
   |              |                  | * Model: .weights `V2, V3`_, `V4`_                       |
   |              |                  | * Run on: CPU or GPU (OpenCV DNN)                        |
   +--------------+------------------+----------------------------------------------------------+
   | YOLO         | YOLO_Ultralytics | * Integrated by linking `ultralytics`_                   |
   |              |                  | * Model: .pt `V3, V5, V8, V9, v10, v11, v12, v26`_       |
   |              |                  | * Run on: CPU or GPU (PyTorch)                           |
   +--------------+------------------+----------------------------------------------------------+

|

.. table:: Supported Trackers

   +--------------+------------------+----------------------------------------------------------+
   | General Name | Config Name      | Details                                                  |
   +==============+==================+==========================================================+
   | Centroid     | Centroid         | * Built-in / Native                                      |
   |              |                  | * Run on: CPU                                            |
   +--------------+------------------+----------------------------------------------------------+
   | SORT         | SORT             | * Integrated by embedding                                |
   |              |                  | * `SORT repo`_                                           |
   |              |                  | * Run on: CPU                                            |
   +--------------+------------------+----------------------------------------------------------+
   | DeepSORT     | DeepSORT         | * Integrated by embedding                                |
   |              |                  | * `DeepSORT repo`_                                       |
   |              |                  | * Run on: CPU or GPU (Tensorflow)                        |
   +--------------+------------------+----------------------------------------------------------+

|

.. table:: Supported ReIDers

   +--------------+------------------+----------------------------------------------------------+
   | General Name | Config Name      | Details                                                  |
   +==============+==================+==========================================================+
   | FaceNet      | FaceNet          | * Integrated by embedding                                |
   |              |                  | * `FaceNet repo`_                                        |
   |              |                  | * Run on: CPU or GPU (Tensorflow)                        |
   +--------------+------------------+----------------------------------------------------------+
   | Torchreid    | Torchreid        | * Integrated by linking `pyppbox-torchreid`_             |
   |              |                  | * Model: OSNet-AIN, OSNet, MLFN                          |
   |              |                  | * Run on: CPU or GPU (PyTorch)                           |
   +--------------+------------------+----------------------------------------------------------+

|

.. _read more here: https://github.com/rathaumons/PoseTReID_DATASET#-posetreid
.. _V2, V3: https://pjreddie.com/darknet/yolo/
.. _V4: https://github.com/AlexeyAB/darknet
.. _ultralytics: https://github.com/numediart/ultralytics-for-vsensebox
.. _V3, V5, V8, V9, v10, v11, v12, v26: https://github.com/ultralytics/assets/releases
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
