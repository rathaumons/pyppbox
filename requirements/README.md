<div align="center">

**[📗 Documentation](https://rathaumons.github.io/pyppbox/) | [🚀 Getting started](https://rathaumons.github.io/pyppbox/getstarted.html) | [💡 Examples](https://rathaumons.github.io/pyppbox/examples.html)**

</div>

## Requirements

| File | Purpose |
| --- | --- |
| `requirements.txt` | Base dependencies |
| `rfdetr.txt` | RF-DETR (Python 3.10+) |
| `faces.txt` | ArcFace/AdaFace on CPU (Python 3.10+) |
| `faces_gpu.txt` | ArcFace/AdaFace on NVIDIA GPU, with CPU fallback |

See [GETSTARTED](../GETSTARTED.md) for installation, model/data wheels, and
switching from CPU to GPU. Install only one face runtime.

## CUDA compatibility

Match ONNX Runtime to your CUDA/cuDNN libraries using the
[compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).
For PyTorch `2.5.1+cu121` (CUDA 12.1, cuDNN 9), use:

```sh
python -m pip install -r requirements/faces_gpu.txt "onnxruntime-gpu==1.20.2"
```

Restart pyppbox after changing runtimes. Requested CUDA initialization failures
raise an error instead of silently falling back to CPU.
