#################################################################################
# Example of exporting an AdaFace embedding model to ONNX
#################################################################################

"""Export CVLFace AdaFace IR18 VGG2 weights using the original AdaFace network.

Run from examples/: python example_16_export_adaface_onnx.py --help.
Download pretrained_model/model.pt from the author's model repository:
https://huggingface.co/minchul/cvlface_adaface_ir18_vgg2
Clone https://github.com/mk-minchul/AdaFace and pass its path below.
Requires torch, onnx and onnxruntime; does not modify the source repository.
The resulting model accepts RGB NCHW floats normalized with (pixel-127.5)/127.5.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adaface-repo', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import onnxruntime as ort
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('adaface_export_network', args.adaface_repo / 'net.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.IR_18((112, 112)).eval()
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    # CVLFace's network prefixes keys with net.; strict loading detects a wrong checkpoint.
    model.load_state_dict({key.removeprefix('net.'): value for key, value in weights.items()}, strict=True)

    class Embeddings(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, image):
            return self.network(image)[0]

    wrapper = Embeddings(model).eval()
    generator = torch.Generator().manual_seed(42)
    example = torch.randn(1, 3, 112, 112, generator=generator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected = wrapper(example).numpy()
        torch.onnx.export(wrapper, example, str(args.output), input_names=['images'],
                          output_names=['embeddings'], opset_version=17,
                          dynamic_axes={'images': {0: 'batch'}, 'embeddings': {0: 'batch'}})
    session = ort.InferenceSession(str(args.output), providers=['CPUExecutionProvider'])
    actual = session.run(None, {'images': example.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-5)
    report = dict(source='minchul/cvlface_adaface_ir18_vgg2/pretrained_model/model.pt',
                  source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  onnx_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  input_color='RGB', input_mean=127.5, input_std=127.5,
                  maximum_export_error=float(np.max(np.abs(expected - actual))))
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
