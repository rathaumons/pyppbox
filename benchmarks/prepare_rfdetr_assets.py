# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Published v1.6.0 RF-DETR data wheel and generated custom-head fixtures."""
import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

from pipeline_cpu import sha256
from prepare_cpu_assets import download

WEIGHTS = {
    'nano': ('rf-detr-nano.pth',
             '22a0a27d0683565f6b6fa86093c6a30385acfa67a33ba63c7111c7cecb329546'),
    'small': ('rf-detr-small.pth',
              '1139dc9379929398962a8935f646f11fd7c3bc57ebfd989e4d96910c6b27eb69'),
    'medium': ('rf-detr-medium.pth',
               'aa81045bfa9e120e8c8324b9437fad08653a861a138da710bf2ef01974a977e6'),
    'large': ('rf-detr-large-2026.pth',
              '0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38'),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--video',type=Path,required=True)
    args=parser.parse_args()
    root=args.directory.resolve(); root.mkdir(parents=True,exist_ok=True)
    wheel_name='pyppbox_data_rfdetr-1.6.0-py3-none-any.whl'
    wheel=root/wheel_name
    download('https://github.com/rathaumons/pyppbox-data/releases/download/v1.6.0/'+wheel_name,
             wheel,'da45c05df2167ae2d414ffcc7e2e659ce9fd18c237c83d59ffddaa897538a4d2')
    assets={}
    # Use the shipped inference checkpoints, extracting only named members of
    # the checksum-verified wheel into the benchmark's own directory.
    with zipfile.ZipFile(wheel) as archive:
        for name,(filename,checksum) in WEIGHTS.items():
            path=root/filename
            with archive.open('pyppbox/data/modules/rf_detr/'+filename) as source, path.open('wb') as output:
                shutil.copyfileobj(source,output)
            if sha256(path)!=checksum:
                raise ValueError(f'Checkpoint checksum mismatch: {filename}')
            assets[name]=dict(path=str(path),sha256=checksum)
    import torch
    with torch.serialization.safe_globals([argparse.Namespace,SimpleNamespace]):
        checkpoint=torch.load(assets['nano']['path'],map_location='cpu',weights_only=True)
    state=checkpoint['model']
    original_classes=state['class_embed.bias'].shape[0]
    for name,value in list(state.items()):
        if 'class_embed' in name and name.endswith(('.weight','.bias')) and value.shape[0]==original_classes:
            # Reuse pretrained person/dog heads at IDs 0/1 and a low background slot.
            # A schema/loading fixture only: no training or quality claim.
            state[name]=value[[1,18,0]].clone()
    original_args=checkpoint.get('args',{})
    config=dict(original_args if isinstance(original_args,dict) else vars(original_args))
    config.update(num_classes=2,class_names=['fixture_person','fixture_dog'])
    custom=root/'custom-two-class.pth'
    torch.save(dict(model=state,args=config),custom)
    assets['custom']=dict(path=str(custom),sha256=sha256(custom),fixture=True,
                          class_names=config['class_names'],source_sha256=assets['nano']['sha256'])
    lightning=root/'custom-lightning.ckpt'
    torch.save(dict(state_dict={'model._orig_mod.'+key:value for key,value in state.items()},
                    hyper_parameters=config),lightning)
    assets['lightning']=dict(path=str(lightning),sha256=sha256(lightning),fixture=True)
    video=args.video.resolve()
    assets['video']=dict(path=str(video),sha256=sha256(video))
    target=root/'assets.json'
    manifest=dict(assets=assets,archive_sha256=sha256(wheel),
                  archive_url='https://github.com/rathaumons/pyppbox-data/releases/tag/v1.6.0')
    target.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(target)


if __name__=='__main__': main()
