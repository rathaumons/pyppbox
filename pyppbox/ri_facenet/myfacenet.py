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


import os
import pickle
import cv2
import numpy as np

import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
tf.autograph.set_verbosity(1)

import skimage.transform

from . import facenet
from . import detect_face

# physical_devices = tf.compat.v1.config.experimental.list_physical_devices('GPU')
# tf.compat.v1.config.experimental.set_memory_growth(physical_devices[0], True)

class MyFacenet(object):


    def __init__(self, cfg):

        self.unk = cfg.mstruct.str.unk_fid
        self.err = cfg.mstruct.str.err_fid
        self.model_file = cfg.model_file
        self.classifier_file = cfg.classifier_file
        self.batch_size = cfg.batch_size
        self.min_confidence = int(100 * cfg.min_confidence)

        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor
        self.margin = 44
        self.image_size = 182
        self.input_image_size = 160

        with tf.Graph().as_default():
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=float(cfg.gpu_mem))
            self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, cfg.model_det)
                self.labels_names_file = os.path.splitext(self.classifier_file)[0] + ".txt"
                with open(self.labels_names_file, 'r') as fp:
                    self.HumanNames = fp.readlines()
                    self.HumanNames = [line.rstrip('\n') for line in self.HumanNames]
                self.HumanNames = sorted(self.HumanNames)
                print("RI Facenet: " + str(self.HumanNames))
                facenet.load_model(self.model_file)
                self.images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]
                self.classifier_file_exp = os.path.expanduser(self.classifier_file)
                with open(self.classifier_file_exp, 'rb') as infile:
                    (self.model, class_names) = pickle.load(infile)
                print('RI Facenet: Ready!')


    def get_face_info(self, bboxes, frame):

        best_class = -1
        best_proba = -1
        nrof_faces = bboxes.shape[0]

        if nrof_faces > 0:
            det = bboxes[:, 0:4]
            cropped = []
            scaled = []
            scaled_reshape = []
            bb = np.zeros((nrof_faces, 4), dtype=np.int32)
            i = 0  # if multiple persons are detected, select first detected one. (shouldn't happen)
            emb_array = np.zeros((1, self.embedding_size))

            bb[i][0] = det[i][0]
            bb[i][1] = det[i][1]
            bb[i][2] = det[i][2]
            bb[i][3] = det[i][3]

            # inner exception
            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                print('RI Facenet: Face is inner of range!')
            else:
                cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                z = len(cropped) - 1
                cropped[z] = facenet.flip(cropped[z], False)
                scaled.append(skimage.transform.resize(cropped[z], (self.image_size, self.image_size), anti_aliasing=True))
                scaled[z] = cv2.resize(scaled[z], (self.input_image_size, self.input_image_size), interpolation=cv2.INTER_CUBIC)
                scaled[z] = facenet.prewhiten(scaled[z])
                scaled_reshape.append(scaled[z].reshape(-1, self.input_image_size, self.input_image_size, 3))
                feed_dict = {self.images_placeholder: scaled_reshape[z], self.phase_train_placeholder: False}
                emb_array[0, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)
                predictions = self.model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                best_class = best_class_indices[0]
                best_proba = int(best_class_probabilities*100)

        return best_class, best_proba


    def recognize_face(self, frame):
        result = " "
        bboxes, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        best_class, best_proba = self.get_face_info(bboxes, frame)
        if best_class != -1 and best_proba != -1:
            if best_proba < self.min_confidence:
                result = self.unk
                print('RI Facenet: Below required confidence!')
            else:
                result = self.HumanNames[best_class] + "_" + str(best_proba) + "%"
                print('RI Facenet: Result --> "%s"' % result)
        else:
            print('RI Facenet: Unable to align')
            result = self.err
        return result


    def recognize_face_by_file(self, path):
        frame = cv2.imread(path)
        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)
        frame = frame[:, :, 0:3]
        return self.recognize_face(frame)

