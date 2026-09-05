# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                           #
#   pyppbox: Toolbox for people detecting, tracking, and re-identifying.    #
#   Copyright (C) 2025 UMONS-Numediart                                      #
#                                                                           #
#   This program is free software: you can redistribute it and/or modify    #
#   it under the terms of the GNU Affero General Public License as          #
#   published by the Free Software Foundation, either version 3 of the      #
#   License, or (at your option) any later version.                         #
#                                                                           #
#   This program is distributed in the hope that it will be useful,         #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#   GNU Affero General Public License for more details.                     #
#                                                                           #
#   You should have received a copy of the GNU Affero General Public License#
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

    """Identify a cropped person using Torchreid embeddings and a saved identity SVM.

    Construction loads the feature extractor even with ``auto_load=False``.
    For direct use, call ``load_classifier()`` before ``recognize()`` or construct
    with ``auto_load=True``. Each instance holds its own extractor and classifier.

    Attributes
    ----------
    extractor : object
        Initialized Torchreid feature extractor on the configured device.
    model : object
        Identity classifier, available after ``load_classifier()``.
    class_names : list[str]
        Class names in pickle order, available after ``load_classifier()``.
    min_confidence : int
        Threshold on a 0-100 scale, computed as ``int(100 * cfg.min_confidence)``.
    auto_load : bool
        Constructor choice to load the classifier; direct ``recognize()`` does not
        perform deferred loading. Deferred loading is handled by the pipeline API.
    """
    def __init__(self, cfg, auto_load=False):
        """Initialize a reider from a populated configuration object.

        Parameters
        ----------
        cfg : pyppbox.config.myconfig.RCFGTorchreid
            Configuration after calling its ``set()`` method, including model paths,
            classifier path, and confidence threshold.
        auto_load : bool
            Defaults to ``False``. Call ``load_classifier()`` during construction when
            True. Otherwise call it explicitly before direct recognition.
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
        """Load or replace the instance's identity classifier and supporting state.

        Read the configured pickle and labels. FaceNet additionally creates its
        TensorFlow session, MTCNN networks, and FaceNet graph; Torchreid's feature
        extractor is already initialized by its constructor. Returns None.
        File, pickle, and model-loading errors propagate to the caller.
        """
        with open(self.classifier_pkl, 'rb') as classifier_file:
            (self.model, self.class_names) = pickle.load(classifier_file)
        add_info_log(f"--------RI : Classifier loaded! <- {getFileName(self.classifier_pkl)}")

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
        best_class = int(np.asarray(best_class_indices[0]).item())
        best_proba = float(np.asarray(best_class_probabilities[0]).item() * 100)
        return best_class, best_proba

    def recognize(self, img, is_bgr=True):
        """Recognize a cropped person in an image.

        Parameters
        ----------
        img : ``numpy.ndarray``
            Nonempty three-channel image crop. Initialize the classifier before calling
            this method directly. The feature extractor resizes the crop.
        is_bgr : bool
            Defaults to ``True``. Convert OpenCV BGR input to RGB when True; False
            means the crop is already RGB.

        Returns
        -------
        str
            Identity label at or above the threshold, the configured unknown ID below
            it, or the configured error ID when no prediction is available.
        float
            Classifier confidence on a 0-100 scale, including below-threshold results;
            0.0 when no prediction is available.

        Notes
        -----
        This method does not catch image-processing or inference exceptions. The
        returned error ID is not a substitute for handling exceptions in direct use.
        """
        result = self.err
        conf = 0.0
        img = self.prepare_image(img, is_bgr=is_bgr)
        best_class, best_proba = self.predict(img)
        if best_class != -1 and best_proba != -1:
            conf = best_proba
            if best_proba < self.min_confidence:
                result = self.unk
                # add_info_log(f"--------RI : Result is below required confidence! -> Return {self.unk}")
            else:
                result = self.class_names[best_class]
                # add_info_log('--------RI : Result = "%s"' % result)
        else:
            # add_warning_log(f"--------RI : The input can't be processed -> Return {self.err}")
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
        """Train an identity SVM from pretrained embeddings and write its files.

        Parameters
        ----------
        C : float
            Defaults to ``1.0``.
            Regularization parameter, passed to sklearn's :code:`SVC(C=C, ...)`.
        kernel : str
            Defaults to ``'rbf'``.
            Embedding-feature kernel, for example ``'linear'``, ``'poly'``, ``'rbf'``, or
            ``'sigmoid'``. This method supplies feature vectors, not a precomputed kernel matrix.
        probability : bool
            Defaults to ``True``.
            Whether to use probability estimates, passed to sklearn's
            :code:`SVC(probability=probability, ...)`.
        decision_function_shape : str
            Defaults to ``'ovr'``.
            Choice of function: :code:`'ovo'` or :code:`'ovr'`, passed to sklearn's
            :code:`SVC(decision_function_shape=decision_function_shape, ...)`.

        Notes
        -----
        The training directory must contain image-only identity subdirectories, with
        at least two identities. This trains the SVM, not the embedding network. It
        overwrites the configured ``.pkl`` and companion ``.txt`` label file and returns
        None. Output directories must exist. Keep ``probability=True`` for classifiers
        used by ``recognize()``, which calls ``predict_proba()``. Call ``load_classifier()``
        after training to use the written classifier in this existing instance.
        """
        dataset = get_dataset(self.train_data)
        paths, labels = get_image_paths_and_labels(dataset)
        add_info_log("--------RI : Extracting features ...")
        emb_array = self.extractor(paths).cpu().numpy()
        add_info_log(f"--------RI : (total_images, features) = {emb_array.shape}")
        add_info_log("--------RI : Training classifier ... ")
        _model = SVC(C=C, kernel=kernel, probability=probability, 
                     decision_function_shape=decision_function_shape)
        _model.fit(emb_array, labels)
        _class_names = [cls.name.replace('_', ' ') for cls in dataset]
        add_info_log(f"--------RI : class_name = {_class_names}")
        with open(self.classifier_pkl, 'wb') as classifier_file:
            pickle.dump((_model, _class_names), classifier_file)
        add_info_log(f"--------RI : Classifier file saved! -> {self.classifier_pkl}")
        classes_txt = self.classifier_pkl[:-3] + "txt"
        with open(classes_txt, 'w') as classes_file:
            classes_file.writelines([str(c) + "\n" for c in _class_names])
        add_info_log(f"--------RI : Classes file saved! -> {classes_txt}")
