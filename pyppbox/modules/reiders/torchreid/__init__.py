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


import cv2
import pickle
import numpy as np
from sklearn.svm import SVC

from pyppbox.utils.commontools import getFileName
from pyppbox.utils.logtools import add_info_log

from .utils import deepreid_extractor, get_dataset, get_image_paths_and_labels


class MyTorchreid(object):

    def __init__(self, cfg, auto_load=False):
        """Initialize according to the given :obj:`cfg` and :obj:`auto_load`.

        Parameters
        ----------
        cfg : RCFGTorchreid
            A :class:`RCFGTorchreid` object which manages the configurations of reidier Torchreid.
        auto_load : bool, optional
            An indication of whether to automatically call :meth:`load_classifier()`.
        """
        self.unk = cfg.unified_strings.unk_did
        self.err = cfg.unified_strings.err_did
        self.mdir = cfg.base_model_path
        self.classifier_pkl = cfg.classifier_pkl
        self.train_data = cfg.train_data
        self.model_name = cfg.model_name
        self.model_path = cfg.model_path
        self.device = cfg.device
        self.min_confidence = int(100 * cfg.min_confidence)
        # add_info_log("--------RI : Initializing ReID model ...")
        self.extractor = deepreid_extractor(self.model_name, self.mdir, 
                                            self.model_path, device=self.device)
        self.auto_load = auto_load
        if self.auto_load:
            self.load_classifier()

    def load_classifier(self):
        """Load the classifier model from the configurations.
        """
        with open(self.classifier_pkl, 'rb') as classifier_file:
            (self.model, self.class_names) = pickle.load(classifier_file)
        add_info_log("--------RI : Classifier loaded! <- " + getFileName(self.classifier_pkl))

    def predict(self, img):
        """
        :meta private:
        """
        best_class = -1
        best_proba = -1
        emb_array = self.extractor(img).cpu().numpy()
        predictions = self.model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        best_class = best_class_indices[0]
        best_proba = float(best_class_probabilities*100)
        return best_class, best_proba

    def recognize(self, img, is_bgr=True):
        """Recognize or re-identify a person in the given :obj:`img`.

        Parameters
        ----------
        img : Mat
            A cv :obj:`Mat` image.
        is_bgr : bool, default=True
            An indication of whether the color channel of given :obj:`img` is BGR.

        Returns
        -------
        str
            A class name.
        float 
            Confidence of the result.
        """
        result = ""
        conf = 100.0
        img = self.prepare_image(img, is_bgr=is_bgr)
        best_class, best_proba = self.predict(img)
        if best_class != -1 and best_proba != -1:
            if best_proba < self.min_confidence:
                result = self.unk
                # add_info_log("--------RI : Result is below required confidence! -> Return " + str(self.unk))
            else:
                result = self.class_names[best_class]
                conf = best_proba
                # add_info_log('-----RI : Result = "%s"' % result)
        else:
            # add_warning_log("--------RI : The input can't be processed -> Return " + str(self.err))
            result = self.err
        return result, conf

    def recognize_file(self, img_path):
        """
        :meta private:
        """
        img = cv2.imread(img_path)
        return self.recognize(img)

    def prepare_image(self, img, is_bgr=True):
        """
        :meta private:
        """
        if is_bgr and img.size != 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def train_classifier(self, C=1.0, kernel='rbf', probability=True, decision_function_shape='ovr'):
        """Train a classifier and dump into pickle .pkl file.

        Parameters
        ----------
        C : float, default=1.0
            Regularization parameter, passed to sklearn's :code:`SVC(C=C, ...)`.
        kernel : str, default='rbf'
            Choice of kernel type: :code:`'linear'`, :code:`'poly'`, :code:`'rbf'`, :code:`'sigmoid'`, 
            or :code:`'precomputed'`, passed to sklearn's :code:`SVC(kernel=kernel, ...)`.
        probability : bool, default=True
            Whether to use probability estimates, passed to sklearn's 
            :code:`SVC(probability=probability, ...)`.
        decision_function_shape : str, default='ovr'
            Choice of function: :code:`'ovo'` or :code:`'ovr'`, passed to sklearn's 
            :code:`SVC(decision_function_shape=decision_function_shape, ...)`.
        """
        dataset = get_dataset(self.train_data)
        paths, labels = get_image_paths_and_labels(dataset)
        add_info_log("--------RI : Extracting features ...")
        emb_array = self.extractor(paths).cpu().numpy()
        add_info_log("--------RI : (total_images, features) = " + str(emb_array.shape))
        add_info_log("--------RI : Training classifier ... ")
        _model = SVC(C=C, kernel=kernel, probability=probability, 
                     decision_function_shape=decision_function_shape)
        _model.fit(emb_array, labels)
        _class_names = [cls.name.replace('_', ' ') for cls in dataset]
        add_info_log("--------RI : class_name = " + str(_class_names))
        with open(self.classifier_pkl, 'wb') as classifier_file:
            pickle.dump((_model, _class_names), classifier_file)
        add_info_log("--------RI : Classifier file saved! -> " + str(self.classifier_pkl))
        classes_txt = self.classifier_pkl[:-3] + "txt"
        with open(classes_txt, 'w') as classes_file:
            classes_file.writelines([str(c) + "\n" for c in _class_names])
        add_info_log("--------RI : Classes file saved! -> " + str(classes_txt))
