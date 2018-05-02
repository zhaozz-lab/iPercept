"""Example architecture."""
from typing import Dict

import tensorflow as tf

from core import BaseDataSource, BaseModel
import util.gaze

data_format = "channels_last"  # Change this to "channels_first" to run on GPU


class SimpleNet(BaseModel):
    """An example neural network architecture."""

    def get_conv2d_multi(self, base_name, n, x, filters, kernel_size, strides, padding='same', data_format='channels_last', activation=tf.nn.relu):
        for i in range(n):
            layer_name = "{}.{}-{}".format(base_name, i, filters)
            x = self.get_conv2d(name=layer_name, x=x, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, activation=activation)
        return x

    def get_conv2d(self, name, x, filters, kernel_size, strides, padding='same', data_format='channels_last', activation=tf.nn.relu):
        with tf.variable_scope(name):
            x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides,
                                         padding=padding, data_format=data_format, activation=activation)

            self.summary.feature_maps('features', x, data_format=data_format)
        return x

    def get_max_pooling2d(self, base_name, x, pool_size, strides, padding, data_format):
        layer_name = "{}-{}-{}".format(base_name, pool_size, strides)
        with tf.variable_scope(layer_name):
            x = tf.layers.max_pooling2d(x, pool_size=pool_size, strides=strides, padding=padding, data_format=data_format)
            self.summary.feature_maps('features', x, data_format=data_format)
        return x

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = input_tensors['eye']
        y = input_tensors['gaze']

        # Trainable parameters should be specified within a known `tf.variable_scope`.
        # This tag is later used to specify the `learning_schedule` which describes when to train
        # which part of the network and with which learning rate.
        #
        # This network has two scopes, 'conv' and 'fc'. Though in practise it makes little sense to
        # train the two parts separately, this is possible.
        with tf.variable_scope('conv'):
            with tf.variable_scope('conv1-20'):
                x = tf.layers.conv2d(x, filters=20, kernel_size=5, strides=1,
                                     padding='same', data_format=data_format, activation=tf.nn.relu)
                self.summary.filters('filters', x)
                self.summary.feature_maps('features', x, data_format=data_format)

            x = self.get_max_pooling2d('maxpool1', x, pool_size=2, strides=2, padding='same', data_format=data_format)

            x = self.get_conv2d_multi("conv2", 1, x, filters=50, kernel_size=5, strides=1, padding='same', data_format=data_format)

            x = self.get_max_pooling2d('maxpool2', x, pool_size=2, strides=2, padding='same', data_format=data_format)

        with tf.variable_scope('fc'):
            # Flatten the 50 feature maps down to one vector
            x = tf.contrib.layers.flatten(x)

            # FC layer
            x = tf.layers.dense(x, units=500, name='fc4', activation=tf.nn.relu)

            self.summary.histogram('fc7/activations', x)

            # Directly regress two polar angles for gaze direction
            x = tf.layers.dense(x, units=2, name='fc8')
            self.summary.histogram('fc8/activations', x)

        # Define outputs
        with tf.variable_scope('mse'):  # To optimize
            loss_terms = {
                'gaze_mse': tf.reduce_mean(tf.squared_difference(x, y)),
            }
        with tf.variable_scope('ang'):  # To evaluate in addition to loss terms
            metrics = {
                'gaze_angular': util.gaze.tensorflow_angular_error_from_pitchyaw(x, y),
            }
        return {'gaze': x}, loss_terms, metrics
