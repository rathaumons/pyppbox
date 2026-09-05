"""Check built artifacts, including exclusion of ignored local planning files."""
import argparse
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


def check_distributions(directory):
    wheels = list(directory.glob('*.whl'))
    sources = list(directory.glob('*.tar.gz'))
    if not wheels or not sources:
        raise AssertionError('Build both a wheel and a source distribution before checking.')
    for artifact in wheels + sources:
        if artifact.suffix == '.whl':
            with zipfile.ZipFile(artifact) as archive:
                names = archive.namelist()
        else:
            with tarfile.open(artifact) as archive:
                names = archive.getnames()
        assert not any('plans' in PurePosixPath(name).parts for name in names), artifact
        for required in ('pyppbox/config/cfg/main.yaml', 'pyppbox/ppb/mt.py', 'LICENSE'):
            assert any(name == required or name.endswith('/' + required) for name in names), (artifact, required)
        print(f'{artifact.name}: package code, default configs, and license present; local plans excluded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    check_distributions(parser.parse_args().directory)
