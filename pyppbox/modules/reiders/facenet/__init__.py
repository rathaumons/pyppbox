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
import pickle
import cv2
import skimage.transform
import numpy as np

from pyppbox.utils.commontools import getFileName, silencer
from pyppbox.utils.logtools import add_info_log, add_warning_log, ignore_this_logger

ignore_this_logger("tensorflow")
ignore_this_logger("facenet")
ignore_this_logger("skimage")
ignore_this_logger("scipy")

import warnings
warnings.filterwarnings("ignore", module="skimage", category=RuntimeWarning) 
warnings.filterwarnings("ignore", module="scipy", category=RuntimeWarning) 

import tensorflow as tf
tf.autograph.set_verbosity(1)

from .origin import facenet as fn
from .origin import detect_face as df


class MyFaceNet(object):

    def __init__(self, cfg, auto_load=False):
        """Initialize according to the given :obj:`cfg` and :obj:`auto_load`.

        Parameters
        ----------
        cfg : RCFGFaceNet
            A :class:`RCFGFaceNet` object which manages the configurations of reidier FaceNet.
        auto_load : bool, optional
            An indication of whether to automatically call :meth:`load_classifier()`.
        """
        self.unk = cfg.unified_strings.unk_fid
        self.err = cfg.unified_strings.err_fid
        self.model_det = cfg.model_det
        self.model_file = cfg.model_file
        self.classifier_file = cfg.classifier_pkl
        self.batch_size = cfg.batch_size
        self.min_confidence = int(100 * cfg.min_confidence)
        self.gpu_mem = cfg.gpu_mem
        self.train_data = cfg.train_data
        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor
        self.margin = 44
        self.image_size = 182
        self.input_image_size = 160
        self.auto_load = auto_load
        if self.auto_load:
            self.load_classifier()

    def load_classifier(self):
        with tf.Graph().as_default():
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=float(self.gpu_mem))
            self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, 
                                                                             allow_soft_placement=True))
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = df.create_mtcnn(self.sess, self.model_det)
                self.labels_names_file = os.path.splitext(self.classifier_file)[0] + ".txt"
                with open(self.labels_names_file, 'r') as fp:
                    self.pnames = fp.readlines()
                    self.pnames = [line.rstrip('\n') for line in self.pnames]
                self.pnames = sorted(self.pnames)
                # add_info_log("--------RI : " + str(self.pnames))
                fn.load_model(self.model_file)
                self.images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]
                self.classifier_file_exp = os.path.expanduser(self.classifier_file)
                with open(self.classifier_file_exp, 'rb') as infile:
                    (self.model, class_names) = pickle.load(infile)
                add_info_log("--------RI : Classifier loaded! <- " + getFileName(self.classifier_file))

    def predict(self, scaled_reshape_img):
        """
        :meta private:
        """
        best_class = -1
        best_proba = -1
        feed_dict = {self.images_placeholder: scaled_reshape_img, self.phase_train_placeholder: False}
        emb_array = np.zeros((1, self.embedding_size))
        emb_array[0, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)
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
        bboxes, _ = df.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        if bboxes.shape[0] > 0:
            scaled_reshape_img = self.make_facenet_image(bboxes, img)
            best_class, best_proba = self.predict(scaled_reshape_img)
            if best_class != -1 and best_proba != -1:
                if best_proba < self.min_confidence:
                    result = self.unk
                    # add_info_log("--------RI : Result is below required confidence! -> Return " + str(self.unk))
                else:
                    result = self.pnames[best_class]
                    conf = best_proba
                    # add_info_log('-----RI : Result -> "%s"' % result)
        else:
            # add_warning_log("--------RI : Can't find any face! -> Return " + str(self.err))
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
            if img.ndim == 2:
                img = fn.to_rgb(img)
            img = img[:, :, 0:3]
        return img

    @silencer
    def make_facenet_image(self, bboxes, img):
        """
        :meta private:
        """
        det = bboxes[:, 0:4]
        bb = np.zeros((1, 4), dtype=np.int32)
        bb[0][0] = det[0][0]
        bb[0][1] = det[0][1]
        bb[0][2] = det[0][2]
        bb[0][3] = det[0][3]
        cropped_img = img[bb[0][1]:bb[0][3], bb[0][0]:bb[0][2], :]
        scaled_img = skimage.transform.resize(cropped_img, (self.image_size, self.image_size), 
                                              anti_aliasing=True)
        scaled_img = cv2.resize(scaled_img, (self.input_image_size, self.input_image_size), 
                                interpolation=cv2.INTER_CUBIC)
        scaled_img = fn.prewhiten(scaled_img)
        scaled_reshape_img = scaled_img.reshape(-1, self.input_image_size, self.input_image_size, 3)
        return scaled_reshape_img

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
        import math
        from sklearn.svm import SVC
        with tf.Graph().as_default():

            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=float(self.gpu_mem))
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
            with sess.as_default():

                dataset = fn.get_dataset(self.train_data)
                classifier_filename_exp = os.path.expanduser(self.classifier_file)

                paths, labels = fn.get_image_paths_and_labels(dataset)
                add_info_log('-----RI : Number of people: %d' % len(dataset))

                add_info_log('-----RI : Loading feature extraction model ... ')
                modeldir = self.model_file
                fn.load_model(modeldir)

                images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                # Run forward pass to calculate embeddings
                add_info_log('-----RI : Calculating features ... ')
                batch_size = self.batch_size
                image_size = 160
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i * batch_size
                    end_index = min((i + 1) * batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = fn.load_data(paths_batch, False, False, image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                # Train classifier
                add_info_log('-----RI : Training classifier ... ')
                model = SVC(C=C, kernel=kernel, probability=probability, 
                            decision_function_shape=decision_function_shape)
                model.fit(emb_array, labels)

                # Create a list of class names & save classifier model
                class_names = [cls.name.replace('_', ' ') for cls in dataset]
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                add_info_log('-----RI : Classifier file saved! -> %s' % classifier_filename_exp)
                classes_txt = classifier_filename_exp[:-3] + "txt"
                with open(classes_txt, 'w') as classes_file:
                    classes_file.writelines([str(c) + "\n" for c in class_names])
                add_info_log("--------RI : Classes file saved! -> " + str(classes_txt))
