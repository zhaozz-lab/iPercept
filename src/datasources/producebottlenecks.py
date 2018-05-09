from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile
import h5py
import cv2
import numpy as np
from PIL import Image
from cv2.cv2 import imshow
from six.moves import urllib
from matplotlib import pyplot as plt
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
# MODEL_INPUT_WIDTH = 299
# MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_WIDTH = 36
MODEL_INPUT_HEIGHT = 60
MODEL_INPUT_DEPTH = 1
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

model_dir = 'inception'
image_dir = 'train'
bottleneck_dir = 'bottlenecks'


def load_images_from_hd5(path_in, key, dataset):
    f = h5py.File(path_in, 'r')
    return f[key][dataset]


def encode_and_bottleneck(i, image, sess, jpeg_data_tensor, bottleneck_tensor):
    if i%100 == 0:
        print(i)
    image_rgb = image
    # image_rgb = np.zeros((36,60,3))
    # for i in range(3):
    #     image_rgb[..., i] = image
    # We need to encode the image as jpg
    image_rgb = cv2.imencode('.jpg', image_rgb)[1].tostring()
    return run_bottleneck_on_image(sess, image_rgb, jpeg_data_tensor, bottleneck_tensor)


def create_bottleneck_values(sess, jpeg_data_tensor, bottleneck_tensor, images):

    bottleneck_values = [encode_and_bottleneck(i, img, sess, jpeg_data_tensor, bottleneck_tensor) for i,img in enumerate(images)]
    return np.array(bottleneck_values)


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def maybe_download_and_extract():
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def create_inception_graph():
    with tf.Session() as sess:
        model_filename = os.path.join(model_dir, 'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (tf.import_graph_def(graph_def, name='',
                                                                                             return_elements=[
                                                                                                 BOTTLENECK_TENSOR_NAME,
                                                                                                 JPEG_DATA_TENSOR_NAME,
                                                                                                 RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def main():
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())

    sess = tf.Session()
    path_in = "../../datasets/MPIIGaze_kaggle_students.h5"
    path_out = "../../datasets/MPIIGaze_kaggle_students_bottleneck.h5"

    file_out = h5py.File(path_out, 'w')


    keys = ['train', 'test', 'validation']

    for key in keys:
        print('Processing key: ', key)
        file_out.create_group(key)

        if key != 'test':
            gaze_data = load_images_from_hd5(path_in, key, 'gaze')
            file_out[key]['gaze'] = np.array(gaze_data)
        #
        head_data = load_images_from_hd5(path_in, key, 'head')
        file_out[key]['head'] = np.array(head_data)

        eye_data = load_images_from_hd5(path_in, key, 'eye')

        eye_data = eye_data[:1000]

        bottleneck_values = create_bottleneck_values(sess, jpeg_data_tensor, bottleneck_tensor, eye_data)

        file_out[key]['eye'] = bottleneck_values




if __name__ == '__main__':


    main()

