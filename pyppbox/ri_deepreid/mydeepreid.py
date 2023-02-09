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


from __future__ import division, print_function, absolute_import

import os
# import sys
import pickle
import numpy as np

import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from ._utils_ import *
from sklearn.svm import SVC


class MyDeepReID(object):

    def __init__(self, cfg):
        self.unk = cfg.unk_did
        self.err = cfg.err_did
        self.mdir = cfg.base_model_path
        self.classes_txt = cfg.classes_txt
        self.classifier_pkl = cfg.classifier_pkl
        self.train_data = cfg.train_data
        self.model_name = cfg.model_name
        self.model_path = cfg.model_path
        self.min_confidence = int(100 * cfg.min_confidence)
        print("RI DeepReID: Initializing ReID model ...")
        self.extractor = set_deep_reider_extractor(self.model_name, self.mdir, self.model_path)

    def load_classifier(self):
        with open(self.classifier_pkl, 'rb') as classifier_file:
            (self.model, self.class_names) = pickle.load(classifier_file)
        print("RI DeepReID: Classifier file loaded! <-- " + self.classifier_pkl)

    def recognize(self, input_img):
        best_class = -1
        best_proba = -1
        emb_array = self.extractor(input_img).cpu().numpy()
        predictions = self.model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        best_class = best_class_indices[0]
        best_proba = int(best_class_probabilities*100)
        return best_class, best_proba

    def recoginize_plus(self, input_img):
        best_class, best_proba = self.recognize(input_img)
        if best_class != -1 and best_proba != -1:
            if best_proba < self.min_confidence:
                result = self.unk
                print('RI DeepReID: Below required confidence!')
            else:
                result = self.class_names[best_class] + "_" + str(best_proba) + "%"
                print('RI DeepReID: Result --> "%s"' % result)
        else:
            print('RI DeepReID: Unable to align')
            result = self.err
        return result

    def train_classifier(self):
        dataset = get_dataset(self.train_data)
        paths, labels = get_image_paths_and_labels(dataset)
        print("RI DeepReID: Extracting features ...")
        emb_array = self.extractor(paths).cpu().numpy()
        print("RI DeepReID: (total_images, features) = " + str(emb_array.shape))
        print("RI DeepReID: Training classifier ... ")
        # _model = SVC(kernel='linear', probability=True)
        _model = SVC(C=1, probability=True, decision_function_shape='ovr')
        _model.fit(emb_array, labels)
        _class_names = [cls.name.replace('_', ' ') for cls in dataset]
        print("RI DeepReID: class_name = " + str(_class_names))
        with open(self.classifier_pkl, 'wb') as classifier_file:
            pickle.dump((_model, _class_names), classifier_file)
        print("RI DeepReID: Classifier file saved! --> " + str(self.classifier_pkl))
        with open(self.classes_txt, 'w') as classes_file:
            classes_file.writelines([str(c) + "\n" for c in _class_names])
        print("RI DeepReID: Classes file saved! --> " + str(self.classes_txt))
