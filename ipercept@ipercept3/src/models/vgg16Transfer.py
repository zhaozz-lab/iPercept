"""VGG Net Tranfer Learning Architecture"""
from typing import Dict

import tensorflow as tf
import numpy as np

import util.gaze
from core import BaseDataSource
from models.custom import CustomModel
import logging


data_format = "channels_last"  # Change this to "channels_first" to run on GPU
logger = logging.getLogger(__name__)


class vgg16Transfer(CustomModel):
    """VGG Net Architecture with loaded weights"""

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors

        logger.info("input_tensors.keys(): "+str(input_tensors.keys()))  # eye, gaze, head

        x = input_tensors['eye']

        # Resize
        x = tf.image.resize_images(
            x,
            [224, 224]
        )

        y = input_tensors['gaze']

        # only augment if training
        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope('augmentation'):
                x, y = self.augment_x(x, y, mode=mode)

        # Trainable parameters should be specified within a known `tf.variable_scope`.
        # This tag is later used to specify the `learning_schedule` which describes when to train
        # which part of the network and with which learning rate.
        #
        # This network has two scopes, 'conv' and 'fc'. Though in practise it makes little sense to
        # train the two parts separately, this is possible.

        self.parameters = []

        with tf.variable_scope('conv'):
            with tf.name_scope('conv1_1') as scope:

                # image has 1 channel while vvgnet trained on 3, thus kernal is different
                # and weights have to change, so I wont't include these in loaded weights.
                kernel = tf.Variable(tf.truncated_normal([3, 3, 1, 64], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv1_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            with tf.name_scope('conv1_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv1_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool1
            self.pool1 = tf.nn.max_pool(self.conv1_2,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='pool1')

            # conv2_1
            with tf.name_scope('conv2_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv2_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv2_2
            with tf.name_scope('conv2_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv2_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool2
            self.pool2 = tf.nn.max_pool(self.conv2_2,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='pool2')

            # conv3_1
            with tf.name_scope('conv3_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv3_2
            with tf.name_scope('conv3_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv3_3
            with tf.name_scope('conv3_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool3
            self.pool3 = tf.nn.max_pool(self.conv3_3,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='pool3')

            # conv4_1
            with tf.name_scope('conv4_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv4_2
            with tf.name_scope('conv4_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv4_3
            with tf.name_scope('conv4_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool4
            self.pool4 = tf.nn.max_pool(self.conv4_3,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='pool4')

            # conv5_1
            with tf.name_scope('conv5_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv5_2
            with tf.name_scope('conv5_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv5_3
            with tf.name_scope('conv5_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool5
            self.pool5 = tf.nn.max_pool(self.conv5_3,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='pool4')

        with tf.variable_scope('fc'):
            # fc1
            with tf.name_scope('fc1') as scope:
                shape = int(np.prod(self.pool5.get_shape()[1:]))
                fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                       dtype=tf.float32,
                                                       stddev=1e-1), name='weights')
                fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                   trainable=True, name='biases')
                pool5_flat = tf.reshape(self.pool5, [-1, shape])
                fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
                self.fc1 = tf.nn.relu(fc1l)
                self.summary.histogram('fc1/activations', self.fc1)
                self.parameters += [fc1w, fc1b]

            # fc2
            with tf.name_scope('fc2') as scope:
                fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                       dtype=tf.float32,
                                                       stddev=1e-1), name='weights')
                fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                   trainable=True, name='biases')
                fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
                self.fc2 = tf.nn.relu(fc2l)
                self.summary.histogram('fc2/activations', self.fc2)
                self.parameters += [fc2w, fc2b]

            # fc3
            with tf.name_scope('fc3') as scope:
                # This one is modified to go from weight matrix of dim 4096x2 (original) to (4096x2) so don't add to
                # the parameters.
                fc3w = tf.Variable(tf.truncated_normal([4096, 2],
                                                       dtype=tf.float32,
                                                       stddev=1e-1), name='weights')
                fc3b = tf.Variable(tf.constant(1.0, shape=[2], dtype=tf.float32),
                                   trainable=True, name='biases')
                self.fc3 = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
                self.summary.histogram('fc3/activations', self.fc3)
                self.parameters += [fc3w, fc3b]

        # Define outputs
        with tf.variable_scope('mse'):  # To optimize
            loss_terms = {
                'gaze_mse': tf.reduce_mean(tf.squared_difference(self.fc3, y)),
            }
        with tf.variable_scope('ang'):  # To evaluate in addition to loss terms
            metrics = {
                'gaze_angular': util.gaze.tensorflow_angular_error_from_pitchyaw(self.fc3, y),
            }
        return {'gaze': self.fc3}, loss_terms, metrics

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)

        keys = sorted(weights.keys())

        # Leaving out the last 2 parameters to load (because they are the wrong shape)
        # Also drop the first weights (because kernel has one 1 image channel here rather
        # than 3).
        for i, k in list(enumerate(keys))[1:-2]:
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))
