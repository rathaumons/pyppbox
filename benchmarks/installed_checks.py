# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Exercise the candidate wheel outside its source tree, read-only on POSIX."""
import argparse
import os
from pathlib import Path
import stat
import subprocess
import sys

from pipeline_cpu import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--tests', type=Path, required=True)
    args = parser.parse_args()
    root = args.package_root.resolve()
    tests = args.tests.resolve()
    if Path.cwd().resolve().is_relative_to(tests.parent):
        parser.error('Run from a neutral directory outside the checkout.')
    sys.path.insert(0, str(root))
    os.environ['PYTHONPATH'] = str(root)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    os.environ['PYPPBOX_DISABLE_FILE_LOG'] = '1'
    os.environ['PYPPBOX_DISABLE_TERMINAL_LOG'] = '1'
    import pyppbox
    package = Path(pyppbox.__file__).resolve().parent
    if not package.is_relative_to(root):
        raise AssertionError(f'Wrong package imported: {package}')
    print('Testing installed package:', package, flush=True)
    before = {str(path.relative_to(package)):sha256(path) for path in package.rglob('*') if path.is_file()}
    permissions = []
    try:
        if os.name == 'posix':
            for path in [*package.rglob('*'), package]:
                permissions.append((path, stat.S_IMODE(path.stat().st_mode)))
                path.chmod(0o555 if path.is_dir() else 0o444)
            print('Native read-only package permissions enabled.', flush=True)
        # Separate processes also isolate identically named core/GUI test modules
        # and mocked backends from real third-party integration imports.
        for directory in (tests, tests / 'integration'):
            subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(directory), '-v'],
                           check=True, timeout=300)
        after = {str(path.relative_to(package)):sha256(path) for path in package.rglob('*') if path.is_file()}
        if before != after:
            raise AssertionError('Installed package contents changed during checks.')
    finally:
        # This is an owned wheel target, never the source checkout or global environment.
        for path, mode in reversed(permissions):
            path.chmod(mode)


if __name__ == '__main__':
    main()
