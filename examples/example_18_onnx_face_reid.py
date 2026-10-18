#################################################################################
# Example of recognizing faces and training an ArcFace or AdaFace classifier
#################################################################################

"""Train or use an ArcFace/AdaFace identity SVM with a JSON/YAML config.

Run from examples/: python example_18_onnx_face_reid.py --config arcface.json --image face.jpg.
Add --train to train the identity SVM first. Training needs identity subfolders.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    from pyppbox.config.configtools import getCFGDict
    from pyppbox.config.myconfig import RCFGArcFace, RCFGAdaFace
    from pyppbox.modules.reiders.onnxface import MyArcFace, MyAdaFace
    document = getCFGDict(str(args.config))
    name = str(document.get('ri_name', '')).lower()
    if name not in ('arcface', 'adaface'):
        parser.error('The configuration must select ArcFace or AdaFace.')
    cls, factory = (RCFGArcFace, MyArcFace) if name == 'arcface' else (RCFGAdaFace, MyAdaFace)
    cfg = cls()
    cfg.set(document)
    backend = factory(cfg)
    try:
        if args.train:
            backend.train_classifier()
        backend.load_classifier()
        label, confidence = backend.recognize_file(str(args.image))
        print(json.dumps(dict(identity=label, svm_confidence_percent=confidence)))
    finally:
        backend.close()


if __name__ == '__main__':
    main()
