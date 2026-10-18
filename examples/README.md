<div align="center">

**[📗 Documentation](https://rathaumons.github.io/pyppbox/) | [🚀 Getting started](https://rathaumons.github.io/pyppbox/getstarted.html) | [💡 Examples](https://rathaumons.github.io/pyppbox/examples.html)**

</div>

Run **all examples from this `examples/` directory**. Install the package, optional
extras, and required model/data wheels first, following the getting-started guide.
Install checkout extras such as `.[rfdetr]` or `.[faces]` from the repository root
before changing directories.

From the repository root:

```sh
cd examples
python example_01_basic.py
```

Examples 1–14 cover the existing configuration, detection, tracking, ReID,
evaluation, direct-module, and multithreading interfaces. Examples 15–18 require
the V4 development checkout on `dev`:

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
`.[rfdetr]` for RF-DETR or `.[faces]` for ArcFace/AdaFace, plus the selected
model's data wheel; see their module guides. Each face ReIDer uses its own
classifier file.

Example 12 overwrites the configured FaceNet and both Torchreid GTA V classifiers.
Use example 18 for ArcFace/AdaFace identity classifiers. Neural-network weights
remain frozen during identity-classifier training.
