# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Pinned RF-DETR 1.10 detection weights and generated custom-head fixtures."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from pipeline_cpu import sha256
from prepare_cpu_assets import download

WEIGHTS = {
    'nano': ('rf-detr-nano.pth','nano_coco/checkpoint_best_regular.pth',
             'd8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d'),
    'small': ('rf-detr-small.pth','small_coco/checkpoint_best_regular.pth',
              'd81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33'),
    'medium': ('rf-detr-medium.pth','medium_coco/checkpoint_best_regular.pth',
               '749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8'),
    'large': ('rf-detr-large-2026.pth','rf-detr-large-2026.pth',
              '0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38'),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--video',type=Path,required=True)
    args=parser.parse_args()
    root=args.directory.resolve(); root.mkdir(parents=True,exist_ok=True)
    assets={}
    for name,(filename,url,checksum) in WEIGHTS.items():
        path=root/filename
        download('https://storage.googleapis.com/rfdetr/'+url,path,checksum)
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
    target.write_text(json.dumps(dict(assets=assets),indent=2)+'\n',encoding='utf-8')
    print(target)


if __name__=='__main__': main()
