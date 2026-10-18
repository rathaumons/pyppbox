"""Resolve the built wheel with each face requirements file without installing packages."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def check_face_dependencies(wheel):
    requirements = Path(__file__).resolve().parents[1] / 'requirements'
    with tempfile.TemporaryDirectory() as directory:
        for filename, wanted, forbidden in (('faces.txt', 'onnxruntime', 'onnxruntime-gpu'),
                                            ('faces_gpu.txt', 'onnxruntime-gpu', 'onnxruntime')):
            report = Path(directory) / (filename + '.json')
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--dry-run', '--ignore-installed',
                 '--report', str(report), str(wheel.resolve()), '-r', str(requirements / filename)],
                capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f'{filename} dependency resolution failed:\n{result.stdout}\n{result.stderr}')
            packages = {entry['metadata']['name'].lower().replace('_', '-'): entry['metadata']['version']
                        for entry in json.loads(report.read_text(encoding='utf-8'))['install']}
            assert wanted in packages, (filename, 'missing runtime', packages)
            assert forbidden not in packages, (filename, 'conflicting runtime', packages)
            assert 'insightface' not in packages, (filename, 'unexpected SDK dependency', packages)
            print(f'{filename}: {len(packages)} dependencies resolved; {wanted}=={packages[wanted]}; '
                  f'no {forbidden} or insightface')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    check_face_dependencies(parser.parse_args().wheel)
