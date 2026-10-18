<div align="center">

**[📗 Documentation](https://rathaumons.github.io/pyppbox/) | [🚀 Getting started](https://rathaumons.github.io/pyppbox/getstarted.html) | [💡 Examples](https://rathaumons.github.io/pyppbox/examples.html)**

</div>

## Optional module requirements

Install pyppbox using the getting-started guide, then install the requirements
for the optional modules you need. Run these commands from the repository root
with Python 3.10 or newer:

```sh
python -m pip install -r requirements/rfdetr.txt  # RF-DETR
```

For ArcFace/AdaFace, choose one runtime:

```sh
python -m pip install -r requirements/faces.txt      # CPU
python -m pip install -r requirements/faces_gpu.txt  # NVIDIA GPU; see runtime setup below
```

These files supplement the base dependencies in `requirements/requirements.txt`.
Model/data wheels are installed separately as described in the getting-started guide.

pyppbox includes its SCRFD inference adapter, adapted from InsightFace under
the MIT license. The InsightFace Python package is not required. A fresh
installation using `faces_gpu.txt` requests only `onnxruntime-gpu`, without CPU
`onnxruntime` or post-install cleanup. `faces.txt` selects the CPU runtime.

### NVIDIA runtime setup

Choose an `onnxruntime-gpu` version that matches your CUDA and cuDNN libraries.
The requirements file does not detect the installed PyTorch/CUDA version. See the
[ONNX Runtime compatibility guide](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

For the tested Windows PyTorch `2.5.1+cu121` setup (CUDA 12.1 and cuDNN 9):

```sh
python -m pip install -r requirements/faces_gpu.txt "onnxruntime-gpu==1.20.2"
```

For other CUDA setups, substitute a compatible GPU version. Use the Python
interpreter that launches pyppbox and restart it after
installation. The GPU runtime also supports `device=cpu`; do not install both
face requirements files to obtain CPU fallback. pyppbox raises if an explicitly requested CUDA
provider fails to initialize.

### Switching from CPU to GPU

Installing `faces_gpu.txt` does not remove an existing CPU runtime. ONNX Runtime
[recommends installing only one runtime package per environment](https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime).
If switching from CPU, or if both packages are installed, close pyppbox and
remove both distributions before reinstalling the GPU runtime:

```sh
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install -r requirements/faces_gpu.txt
```

The packages share runtime files. Removing both first ensures the GPU package is
installed again completely, even if pip previously considered it installed.
Apply any CUDA-compatible version pin to the reinstall command too, as in the
CUDA 12.1 example above. Restart pyppbox after installation. This removal step is
unnecessary for a fresh environment.
