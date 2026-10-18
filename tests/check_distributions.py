"""Check package contents, dependency bounds, and retained license notices."""
import argparse
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

from packaging.requirements import Requirement


def check_distributions(directory):
    wheels = list(directory.glob('*.whl'))
    sources = list(directory.glob('*.tar.gz'))
    if not wheels or not sources:
        raise AssertionError('Build both a wheel and a source distribution before checking.')
    for artifact in wheels + sources:
        is_wheel = artifact.suffix == '.whl'
        with zipfile.ZipFile(artifact) if is_wheel else tarfile.open(artifact) as archive:
            names = archive.namelist() if is_wheel else archive.getnames()

            def read_file(name):
                if is_wheel:
                    return archive.read(name)
                with archive.extractfile(name) as member:
                    return member.read()

            def find_file(suffix):
                matches = [name for name in names if name == suffix or name.endswith('/' + suffix)]
                assert len(matches) == 1, (artifact, suffix, matches)
                return matches[0]

            assert not any('plans' in PurePosixPath(name).parts for name in names), artifact
            for required in ('pyppbox/config/cfg/main.yaml', 'pyppbox/config/torchreidmodels.py',
                             'pyppbox/config/validation.py', 'pyppbox/utils/batching.py', 'pyppbox/utils/runtime.py',
                             'pyppbox/modules/reiders/torchreid/model_dict.py', 'pyppbox/ppb/mt.py',
                             'pyppbox/modules/detectors/rfdetr/__init__.py', 'pyppbox/gui/processes.py',
                             'pyppbox/modules/reiders/onnxface/__init__.py',
                             'pyppbox/modules/reiders/onnxface/scrfd.py', 'pyppbox/gui/ui_onnxface.py'):
                find_file(required)

            # Check complete license/notice contents, not just their filenames.
            repo = Path(__file__).resolve().parents[1]
            for source in ('LICENSE', 'NOTICE', 'licenses/GPL-3.0.txt', 'licenses/InsightFace-MIT.txt'):
                packaged = read_file(find_file(PurePosixPath(source).name)).decode('utf-8')
                expected = (repo / source).read_text(encoding='utf-8')
                assert packaged.replace('\r\n', '\n') == expected, (artifact, source)
            assert (repo / 'LICENSE').read_text(encoding='utf-8').lstrip().startswith('GNU AFFERO GENERAL PUBLIC LICENSE')

            metadata_name = find_file('METADATA') if is_wheel else next(
                name for name in names if len(PurePosixPath(name).parts) == 2 and name.endswith('/PKG-INFO')
            )
            metadata = BytesParser().parsebytes(read_file(metadata_name))
            assert 'Programming Language :: Python :: 3.13' in metadata.get_all('Classifier', []), artifact
            requirements = [Requirement(value) for value in metadata.get_all('Requires-Dist', [])]
            yolo = [requirement for requirement in requirements if requirement.name == 'pyppbox-ultralytics']
            assert len(yolo) == 1 and yolo[0].marker is None, (artifact, yolo)
            assert '8.4.144' in yolo[0].specifier, (artifact, yolo)
            assert not any(requirement.name in ('vsensebox-ultralytics', 'ultralytics')
                           for requirement in requirements), artifact
            assert not metadata.get_all('Provides-Extra', []), artifact
            optional_names = {'rfdetr', 'insightface', 'onnxruntime', 'onnxruntime-gpu'}
            assert not any(requirement.name in optional_names for requirement in requirements), artifact
            for filename, runtime in (('rfdetr.txt', 'rfdetr'), ('faces.txt', 'onnxruntime'),
                                      ('faces_gpu.txt', 'onnxruntime-gpu')):
                source = 'requirements/' + filename
                expected = [Requirement(line.strip()) for line in (repo / source).read_text().splitlines()
                            if line.strip() and not line.lstrip().startswith('#')]
                assert len(expected) == 1 and expected[0].name == runtime, (artifact, filename, expected)
                if runtime == 'rfdetr':
                    assert '1.10.0' in expected[0].specifier and '1.11.0' not in expected[0].specifier
                if not is_wheel:
                    packaged = read_file(find_file(source)).decode('utf-8')
                    assert packaged.replace('\r\n', '\n') == (repo / source).read_text(), (artifact, source)
            if not is_wheel:
                find_file('requirements/requirements.txt')
            opencv = [requirement for requirement in requirements if requirement.name == 'opencv-contrib-python']
            assert len(opencv) == 1, (artifact, opencv)
            # Installation must retain Darknet support, including on future major releases.
            for version in ('4.11.0', '4.13.0'):
                assert version in opencv[0].specifier, (artifact, opencv[0], version)
            for version in ('5.0.0', '5.1.0', '6.0.0'):
                assert version not in opencv[0].specifier, (artifact, opencv[0], version)
            assert (metadata['License-Expression'] or metadata['License']) == 'AGPL-3.0-or-later', artifact
            license_files = metadata.get_all('License-File', [])
            assert {PurePosixPath(name).name for name in license_files} == {
                'LICENSE', 'NOTICE', 'GPL-3.0.txt', 'InsightFace-MIT.txt'
            }, (artifact, license_files)

            launcher = read_file(find_file('pyppbox/gui/ui_launcher.py')).decode('utf-8')
            assert 'PYPPBOX (AGPLV3+)' in launcher and 'PYPPBOX (GPLV3+)' not in launcher, artifact
            headers = read_file(find_file('pyppbox/config/myconfig.py')).decode('utf-8')
            short_notice = 'Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License'
            assert headers.startswith('# ' + short_notice), artifact
            assert ':: ' + short_notice in headers and 'GNU General Public License' not in headers, artifact

            for source in (
                'pyppbox/modules/trackers/sort/origin/sort.py',
                'pyppbox/modules/reiders/facenet/origin/align_dataset_mtcnn.py',
                'pyppbox/modules/reiders/facenet/origin/detect_face.py',
                'pyppbox/modules/reiders/facenet/origin/facenet.py',
            ):
                packaged = read_file(find_file(source)).decode('utf-8')
                assert packaged.replace('\r\n', '\n') == (repo / source).read_text(encoding='utf-8'), (artifact, source)

        print(f'{artifact.name}: code/configs, base-only dependencies, optional requirements files, default YOLO backend, OpenCV bound, AGPL metadata, and third-party notices verified; local plans excluded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    check_distributions(parser.parse_args().directory)
