<div align="center">

**[📗 Documentation](https://rathaumons.github.io/pyppbox/) | [🚀 Getting started](https://rathaumons.github.io/pyppbox/getstarted.html) | [💡 Examples](https://rathaumons.github.io/pyppbox/examples.html)**

</div>

Run **all examples from this `examples/` directory**. Install the package, optional
module requirements, and required model/data wheels first, following the getting-started guide.
Install optional requirements with `python -m pip install -r requirements/rfdetr.txt`
or `python -m pip install -r requirements/faces.txt` from the repository root
before changing directories.

From the repository root:

```sh
cd examples
python example_01_basic.py
```

Examples 1–14 cover configuration, detection, tracking, ReID, evaluation,
direct-module, and multithreading interfaces. Example 12 also includes the V4
ArcFace/AdaFace reiders. Examples 15–18 cover additional V4 workflows:

| Example | Purpose | Run from `examples/` |
| --- | --- | --- |
| 15 | RF-DETR detection with SORT tracking | `python example_15_rfdetr.py --device cpu --frames 30` |
| 16 | Export and verify an AdaFace ONNX model | `python example_16_export_adaface_onnx.py --help` |
| 17 | Compare FaceNet, ArcFace, and AdaFace | `python example_17_benchmark_faces.py --help` |
| 18 | Recognize faces or train an ArcFace/AdaFace classifier | `python example_18_onnx_face_reid.py --help` |

The command-line examples accept explicit checkpoint, dataset, and output paths;
see their documentation pages for complete commands and prerequisites. Relative
paths are resolved from the current `examples/` working directory.

For examples 3, 4, 5, 11, and 13, replace the `C:/pyppbox_v3/` placeholder in the
script or its custom config files with your installed `pyppbox` directory. Find it
with `python -c "import os, pyppbox; print(os.path.dirname(pyppbox.__file__))"`.

The `cfg/` directory used by example 3 includes all supported modules, including
`RF_DETR`, `ArcFace`, and `AdaFace`. Select modules in `cfg/main.yaml` and edit
their parameter blocks in `cfg/detectors.yaml`, `cfg/trackers.yaml`, or
`cfg/reiders.yaml`. The new modules use CPU in these sample configs. Install
`requirements/rfdetr.txt` for RF-DETR or `requirements/faces.txt` for ArcFace/AdaFace
using `python -m pip install -r` from the repository root, plus the selected
model's data wheel; see their module guides. Each face ReIDer uses its own
classifier file.

Example 12 retrains five GTA V identity classifiers: FaceNet, Torchreid OSNet-AIN,
Torchreid OSNet, ArcFace, and AdaFace. It overwrites their configured classifier
files; change the output paths to keep the supplied files, or remove entries from
the training loop to select specific reiders. The two ONNX reiders default to CPU;
install their optional runtime and data wheels first. Example 18 demonstrates
training and recognizing a face with a custom ArcFace/AdaFace configuration.
Neural-network weights remain frozen during identity-classifier training.
