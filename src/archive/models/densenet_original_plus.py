"""DenseNet architecture."""
from typing import Dict


import tensorflow as tf

from core import BaseDataSource

import util.gaze
from models.custom import CustomModel

data_format = "channels_last"  # Change this to "channels_first" to run on GPU


class DenseNetOriginalPlus(CustomModel):
    """An implementation of the DenseNet architecture."""

    def get_optimizer(self, spec):
        return tf.train.GradientDescentOptimizer(
            learning_rate=spec['learning_rate'],
        )

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""

        block_sizes = [6, 12, 64, 48]
        growth_rate = 32

        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = input_tensors['eye']
        y = input_tensors['gaze']

        def bottleneck(l, k, name):
            """

            :param name:
            :param l:
            :param k: growth rate
            :return: layer with reduced number of feature maps: 4*k
            """
            return conv('{}_bottleneck'.format(name), l, k * 4, stride=1, kernel_size=1)

        def conv(name, l, filters, stride, kernel_size=3):
            output=tf.layers.conv2d(l, filters=filters, kernel_size=kernel_size, strides=stride,
                                    padding='same', name=name, data_format=data_format)

            return output

        def add_layer(name, l):
            with tf.variable_scope(name):
                # Using a bottleneck layer to reduce number of input featuremaps
                c = bottleneck(l, growth_rate, name=name)
                c = tf.layers.batch_normalization(c, name='bn1')
                c = tf.nn.relu(c)
                c = conv('{}_conv1'.format(name), c, growth_rate, stride=1, kernel_size=3)
                l = tf.concat([c, l], 3)
            return l

        def add_transition(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                l = bottleneck(l, growth_rate, name)
                l = tf.layers.batch_normalization(l, name='bn1')
                l = tf.nn.relu(l)
                l = tf.layers.conv2d(l, filters=in_channel, strides=1, kernel_size=1, padding='same',
                                     data_format=data_format, name='conv1')
                # changed from tensorpack
                layer = tf.layers.AveragePooling2D(name='pool', padding='same', strides=2,
                                                   pool_size=2, data_format=data_format)
                l = layer.apply(l, scope=tf.get_variable_scope())
            return l

        def global_average_pooling(x, data_format='channels_last', name=None):
            """
            Global average pooling as in the paper `Network In Network
            <http://arxiv.org/abs/1312.4400>`_.
            Args:
                x (tf.Tensor): a 4D tensor.
            Returns:
                tf.Tensor: a NC tensor named ``output``.
            """
            assert x.shape.ndims == 4
            axis = [1, 2] if data_format == 'channels_last' else [2, 3]
            return tf.reduce_mean(x, axis, name=name)

        def dense_net(name):
            with tf.variable_scope('denseblocks'):
                l = conv('conv0', x, filters=2 * growth_rate, kernel_size=3, stride=1)  # original: kernelsize=7, stride=2, but we have much smaller images

                for i in range(len(block_sizes)):
                    scope_name = "block{}_{}".format(i, block_sizes[i])
                    with tf.variable_scope(scope_name):
                        for j in range(block_sizes[i]):
                            l = add_layer("dense_layer.{}".format(j), l)

                        self.summary.feature_maps('features', l, data_format=data_format)
                        l = add_transition("transition.{}".format(i), l)

            with tf.variable_scope('regression'):
                l = tf.layers.batch_normalization(l, name='bnlast')
                l = tf.nn.relu(l)
                l = global_average_pooling(name='gap', x=l, data_format=data_format)
                self.summary.histogram('average_pooling', l)
                l = tf.layers.dense(l, units=1000, activation=tf.nn.tanh)
                self.summary.histogram('dense1000', x)
                regressed_output = tf.layers.dense(l, units=2, name='fc4', activation=None)
                self.summary.histogram('output', x)

            return regressed_output

        output = dense_net('dense_net')

        with tf.variable_scope('mse'):  # To optimize
            loss_terms = {
                'gaze_mse': tf.reduce_mean(tf.squared_difference(output, y)),
            }
        with tf.variable_scope('ang'):  # To evaluate in addition to loss terms
            metrics = {
                'gaze_angular': util.gaze.tensorflow_angular_error_from_pitchyaw(output, y),
            }
        return {'gaze': output}, loss_terms, metrics


