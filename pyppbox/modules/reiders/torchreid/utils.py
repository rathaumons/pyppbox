# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                           #
#   pyppbox: Toolbox for people detecting, tracking, and re-identifying.    #
#   Copyright (C) 2022 UMONS-Numediart                                      #
#                                                                           #
#   This program is free software: you can redistribute it and/or modify    #
#   it under the terms of the GNU General Public License as published by    #
#   the Free Software Foundation, either version 3 of the License, or       #
#   (at your option) any later version.                                     #
#                                                                           #
#   This program is distributed in the hope that it will be useful,         #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#   GNU General Public License for more details.                            #
#                                                                           #
#   You should have received a copy of the GNU General Public License       #
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.  #
#                                                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import os


from pyppbox.utils.commontools import silencer
from pyppbox.utils.logtools import ignore_this_logger, add_error_log


class ImageClass():

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def get_dataset(path):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        datadir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(datadir)
        dataset.append(ImageClass(class_name, image_paths))
    return dataset

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

def get_image_paths(datadir):
    image_paths = []
    if os.path.isdir(datadir):
        images = os.listdir(datadir)
        image_paths = [os.path.join(datadir, img) for img in images]
    return image_paths

@silencer
def deepreid_extractor(model_name, model_dir, model_path, device='cuda'):
    ignore_this_logger("torchreid")
    ignore_this_logger("pyppbox_torchreid")
    from pyppbox_torchreid.utils import FeatureExtractor
    extractor = []
    try:
        extractor = FeatureExtractor(
            base_model_name = model_name,
            base_model_dir = model_dir,
            model_path = model_path,
            device = device
        )
    except Exception as e:
        msg = "deepreid_extractor() -> " + str(e)
        add_error_log(msg)
        raise ValueError(msg)
    return extractor
