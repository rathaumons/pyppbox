"""
    pyppbox: Toolbox for people detecting, tracking, and re-identifying.
    Copyright (C) 2022 UMONS-Numediart

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import pickle

from _utils_ import *
from sklearn.svm import SVC
from train_config import CFGDeepReID

cfg = CFGDeepReID()

classes_txt = cfg.classes_txt
classifier_pkl = cfg.classifier_pkl
train_data = cfg.train_data
model_name = cfg.model_name
model_path = cfg.model_path

current_dir = os.path.dirname(__file__)
base_model_dir = os.path.join(current_dir, 'pretrained/base')
dataset = get_dataset(train_data)

paths, labels = get_image_paths_and_labels(dataset)

print("RI DeepReID: Extracting features ...")
extractor = set_deep_reider_extractor(model_name, base_model_dir, model_path)
emb_array = extractor(paths).cpu().numpy()
print("RI DeepReID: (total_images, features) = " + str(emb_array.shape))

print("RI DeepReID: Training classifier ... ")
_model = SVC(C=1, probability=True, decision_function_shape='ovr')
_model.fit(emb_array, labels)
_class_names = [cls.name.replace('_', ' ') for cls in dataset]
print("RI DeepReID: class_name = " + str(_class_names))

with open(classifier_pkl, 'wb') as classifier_file:
    pickle.dump((_model, _class_names), classifier_file)
print("RI DeepReID: Classifier file saved! --> " + str(classifier_pkl))

with open(classes_txt, 'w') as classes_file:
    classes_file.writelines([str(c) + "\n" for c in _class_names])
print("RI DeepReID: Classes file saved! --> " + str(classes_txt))
