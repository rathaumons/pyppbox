import os
import math
import pickle
import facenet
import numpy as np

import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
tf.autograph.set_verbosity(1)

from sklearn.svm import SVC
from train_config import CFGFacenet

print("Initializing ... ")
myconfig = CFGFacenet()

with tf.Graph().as_default():

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=float(myconfig.gpu_mem))
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

    with sess.as_default():

        dataset = facenet.get_dataset(myconfig.data_path)
        classifier_filename_exp = os.path.expanduser(myconfig.classifier_file)

        paths, labels = facenet.get_image_paths_and_labels(dataset)
        print('Number of people: %d' % len(dataset))

        print('Loading feature extraction model ... ')
        modeldir = myconfig.model_file
        facenet.load_model(modeldir)

        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # Run forward pass to calculate embeddings
        print('Calculating features ... ')
        batch_size = myconfig.batch_size
        image_size = 160
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

        # Train classifier
        print('Training classifier ... ')
        # model = SVC(kernel='linear', probability=True)
        # model = SVC(kernel='rbf', probability=True, decision_function_shape='ovo')
        model = SVC(C=1, probability=True, decision_function_shape='ovr')
        model.fit(emb_array, labels)

        # Create a list of class names & save classifier model
        class_names = [cls.name.replace('_', ' ') for cls in dataset]
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Classifier model file saved: "%s"' % classifier_filename_exp)
