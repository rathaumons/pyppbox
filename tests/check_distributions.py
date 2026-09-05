"""Check package contents, AGPL metadata, and retained third-party notices."""
import argparse
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


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
            for required in ('pyppbox/config/cfg/main.yaml', 'pyppbox/ppb/mt.py'):
                find_file(required)

            # Check complete license/notice contents, not just their filenames.
            repo = Path(__file__).resolve().parents[1]
            for source in ('LICENSE', 'NOTICE', 'licenses/GPL-3.0.txt'):
                packaged = read_file(find_file(PurePosixPath(source).name)).decode('utf-8')
                expected = (repo / source).read_text(encoding='utf-8')
                assert packaged.replace('\r\n', '\n') == expected, (artifact, source)
            assert (repo / 'LICENSE').read_text(encoding='utf-8').lstrip().startswith('GNU AFFERO GENERAL PUBLIC LICENSE')

            metadata_name = find_file('METADATA') if is_wheel else next(
                name for name in names if len(PurePosixPath(name).parts) == 2 and name.endswith('/PKG-INFO')
            )
            metadata = BytesParser().parsebytes(read_file(metadata_name))
            assert (metadata['License-Expression'] or metadata['License']) == 'AGPL-3.0-or-later', artifact
            license_files = metadata.get_all('License-File', [])
            assert {PurePosixPath(name).name for name in license_files} == {'LICENSE', 'NOTICE', 'GPL-3.0.txt'}, (artifact, license_files)

            launcher = read_file(find_file('pyppbox/gui/ui_launcher.py')).decode('utf-8')
            assert 'PYPPBOX (AGPLV3+)' in launcher and 'PYPPBOX (GPLV3+)' not in launcher, artifact
            headers = read_file(find_file('pyppbox/config/myconfig.py')).decode('utf-8')
            assert 'GNU Affero General Public License' in headers and 'GNU General Public License' not in headers, artifact

            for source in (
                'pyppbox/modules/trackers/sort/origin/sort.py',
                'pyppbox/modules/reiders/facenet/origin/align_dataset_mtcnn.py',
                'pyppbox/modules/reiders/facenet/origin/detect_face.py',
                'pyppbox/modules/reiders/facenet/origin/facenet.py',
            ):
                packaged = read_file(find_file(source)).decode('utf-8')
                assert packaged.replace('\r\n', '\n') == (repo / source).read_text(encoding='utf-8'), (artifact, source)

        print(f'{artifact.name}: code/configs, AGPL metadata, and third-party notices verified; local plans excluded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    check_distributions(parser.parse_args().directory)
