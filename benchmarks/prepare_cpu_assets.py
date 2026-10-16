# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Download checksum-pinned public test assets outside installed packages."""
import argparse
import json
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen
import zipfile

from pipeline_cpu import sha256


def download(url, destination, expected):
    if destination.is_file() and sha256(destination) == expected:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.part')
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'pyppbox-cpu-benchmark'}), timeout=60) as source:
                with temporary.open('wb') as output:
                    shutil.copyfileobj(source, output)
            if sha256(temporary) != expected:
                raise ValueError(f'Checksum mismatch: {url}')
            temporary.replace(destination)
            return
        except (OSError, ValueError):
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    wheel_name = 'pyppbox_data_torchreid-1.4.0-py3-none-any.whl'
    wheel = directory / wheel_name
    download('https://github.com/rathaumons/pyppbox-data/releases/download/v1.4.0/' + wheel_name,
             wheel, 'a52382ac0dbd1303cb1d34449a5f41fc62cc74889134e52feb75203d37ae709b')
    detector = directory / 'yolov8n-pose.pt'
    download('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt',
             detector, 'c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212')
    entries = {
        'embedding': 'torchreid/models/torchreid/osnet_ain_ms_d_c.pth.tar',
        'classifier': 'torchreid/classifier/gta5_osnet_ain_ms_d_c.pkl',
        'base': 'torchreid/models/base/osnet_ain_x1_0_imagenet.pth',
    }
    paths = {'video': args.video.resolve(), 'detector': detector}
    # Extract only explicit members of the verified archive; never install assets
    # into site-packages or execute archive paths/metadata.
    with zipfile.ZipFile(wheel) as archive:
        for key, suffix in entries.items():
            members = [name for name in archive.namelist() if name.endswith('/' + suffix)]
            if len(members) != 1:
                raise ValueError(f'Expected exactly one asset: {suffix}: {members}')
            destination = directory / suffix
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(members[0]) as source, destination.open('wb') as output:
                shutil.copyfileobj(source, output)
            paths[key] = destination
    manifest = dict(assets={key:dict(path=str(path), sha256=sha256(path)) for key,path in paths.items()},
                    archive_sha256=sha256(wheel), archive_url='https://github.com/rathaumons/pyppbox-data/releases/tag/v1.4.0')
    target = directory / 'assets.json'
    target.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
